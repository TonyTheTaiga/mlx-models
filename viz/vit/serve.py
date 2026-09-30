"""Serve the visualizer with caching disabled, so edits show up on a plain reload."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = partial(NoCacheHandler, directory=str(Path(__file__).parent / "web"))
    print(f"Lumen on http://localhost:{args.port}")
    ThreadingHTTPServer(("", args.port), handler).serve_forever()
