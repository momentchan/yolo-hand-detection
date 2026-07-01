"""
YOLO hand-detection webcam -> WebSocket broadcaster for sketch_fish.

Run from a clone of https://github.com/cansik/yolo-hand-detection
(after downloading models into models/), or pass --yolo-dir to that repo.

Example:
  pip install -r requirements.txt
  python demo_webcam_ws.py -d 0

The aquarium app listens on ws://127.0.0.1:8765 by default.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from pathlib import Path

import cv2

DEFAULT_WS_HOST = "127.0.0.1"
DEFAULT_WS_PORT = 8765


class DetectionWebSocketHub:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._clients: set[object] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve_forever())

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    async def _serve_forever(self) -> None:
        import websockets

        async def handler(websocket: object) -> None:
            self._clients.add(websocket)
            try:
                await websocket.wait_closed()  # type: ignore[attr-defined]
            finally:
                self._clients.discard(websocket)

        async with websockets.serve(handler, self._host, self._port):
            print(f"[hand-detector] WebSocket listening on ws://{self._host}:{self._port}")
            await asyncio.Future()

    def broadcast(self, payload: dict) -> None:
        if not self._loop or not self._clients:
            return

        message = json.dumps(payload)

        async def send_all() -> None:
            stale = []
            for client in list(self._clients):
                try:
                    await client.send(message)  # type: ignore[attr-defined]
                except Exception:
                    stale.append(client)
            for client in stale:
                self._clients.discard(client)

        asyncio.run_coroutine_threadsafe(send_all(), self._loop)


def load_yolo(network: str, yolo_dir: Path):
    sys.path.insert(0, str(yolo_dir))
    from yolo import YOLO  # noqa: WPS433

    models = yolo_dir / "models"
    if network == "normal":
        return YOLO(str(models / "cross-hands.cfg"), str(models / "cross-hands.weights"), ["hand"])
    if network == "prn":
        return YOLO(
            str(models / "cross-hands-tiny-prn.cfg"),
            str(models / "cross-hands-tiny-prn.weights"),
            ["hand"],
        )
    if network == "v4-tiny":
        return YOLO(
            str(models / "cross-hands-yolov4-tiny.cfg"),
            str(models / "cross-hands-yolov4-tiny.weights"),
            ["hand"],
        )
    return YOLO(str(models / "cross-hands-tiny.cfg"), str(models / "cross-hands-tiny.weights"), ["hand"])


def results_to_payload(
    results: list,
    frame_width: int,
    frame_height: int,
    mirror: bool,
    hand_limit: int,
) -> dict:
    ordered = sorted(results, key=lambda item: item[2], reverse=True)
    if hand_limit != -1:
        ordered = ordered[:hand_limit]

    detections = []
    for _id, _name, confidence, x, y, w, h in ordered:
        detections.append(
            {
                "xmin": int(x),
                "ymin": int(y),
                "xmax": int(x + w),
                "ymax": int(y + h),
                "confidence": float(confidence),
            }
        )

    return {
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "mirror": mirror,
        "detections": detections,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO hand webcam -> WebSocket")
    parser.add_argument(
        "-n",
        "--network",
        default="v4-tiny",
        choices=["normal", "tiny", "prn", "v4-tiny"],
        help="YOLO network type",
    )
    parser.add_argument("-d", "--device", type=int, default=0, help="Webcam device index")
    parser.add_argument("-s", "--size", default=416, type=int, help="YOLO input size")
    parser.add_argument("-c", "--confidence", default=0.2, type=float, help="YOLO confidence")
    parser.add_argument(
        "-nh",
        "--hands",
        default=-1,
        type=int,
        help="Max hands per frame (-1 for all)",
    )
    parser.add_argument("--host", default=DEFAULT_WS_HOST, help="WebSocket host")
    parser.add_argument("--port", default=DEFAULT_WS_PORT, type=int, help="WebSocket port")
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror normalized X (typical for front-facing webcam)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show local OpenCV preview window",
    )
    parser.add_argument(
        "--yolo-dir",
        type=Path,
        default=Path.cwd(),
        help="Path to cansik/yolo-hand-detection checkout (with yolo.py and models/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yolo_dir = args.yolo_dir.resolve()

    if not (yolo_dir / "yolo.py").exists():
        raise SystemExit(f"yolo.py not found in {yolo_dir}. Pass --yolo-dir to the clone.")

    print(f"loading yolo ({args.network}) from {yolo_dir}...")
    yolo = load_yolo(args.network, yolo_dir)
    yolo.size = int(args.size)
    yolo.confidence = float(args.confidence)

    hub = DetectionWebSocketHub(args.host, args.port)
    hub.start()

    print("starting webcam...")
    capture = cv2.VideoCapture(args.device)
    if not capture.isOpened():
        raise SystemExit(f"Could not open webcam device {args.device}")

    if args.preview:
        cv2.namedWindow("preview")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_height, frame_width = frame.shape[:2]
            _width, _height, inference_time, results = yolo.inference(frame)

            payload = results_to_payload(
                results,
                frame_width,
                frame_height,
                args.mirror,
                args.hands,
            )
            hub.broadcast(payload)

            if args.preview:
                fps = 0.0 if inference_time <= 0 else round(1 / inference_time, 2)
                cv2.putText(
                    frame,
                    f"{fps} FPS | hands: {len(payload['detections'])}",
                    (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
                for det in payload["detections"]:
                    cv2.rectangle(
                        frame,
                        (det["xmin"], det["ymin"]),
                        (det["xmax"], det["ymax"]),
                        (0, 255, 255),
                        2,
                    )
                cv2.imshow("preview", frame)
                if cv2.waitKey(1) == 27:
                    break
    finally:
        capture.release()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
