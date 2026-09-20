import argparse
from pathlib import Path
from threading import Thread

from sign_edge.app import create_app
from sign_edge.frame_store import LatestFrameStore
from sign_edge.processor import RichModelProcessor
from sign_edge.viewer import run_viewer


def parse_args():
    parser = argparse.ArgumentParser(description="ESP32 edge frame service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--board-text-url")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = LatestFrameStore()
    processor = RichModelProcessor(args.model) if args.model else None
    if args.gui:
        Thread(target=run_viewer, args=(store,), daemon=True).start()
    app = create_app(
        store=store,
        processor=processor,
        gui_enabled=args.gui,
        board_text_url=args.board_text_url,
    )
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
