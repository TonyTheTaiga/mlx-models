"""Serve a static site (Lumen by default) with caching disabled, so edits show on a plain reload."""

import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    parser.add_argument(
        "--dir", type=Path, default=Path(__file__).parent / "web", help="site to serve"
    )
    args = parser.parse_args()
    handler = partial(NoCacheHandler, directory=str(args.dir))
    print(f"Serving {args.dir} on http://localhost:{args.port}")
    ThreadingHTTPServer(("", args.port), handler).serve_forever()
