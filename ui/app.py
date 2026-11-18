"""Serve the evaluator UI on Laptop 2 using only the Python standard library."""
import argparse
import os
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
ASSETS = {
    "/": ("index.html", "text/html"),
    "/index.html": ("index.html", "text/html"),
    "/style.css": ("style.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        resource = ASSETS.get(urlsplit(self.path).path)
        if resource is None:
            self.send_error(404)
            return
        name, kind = resource
        payload = (ROOT / name).read_text(encoding="utf-8")
        if name == "index.html":
            url = escape(os.getenv("PATHSNAP_API_URL", "http://localhost:8000"), quote=True)
            payload = payload.replace('value="http://localhost:8000"', f'value="{url}"')
        body = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"PathSnap UI: http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
