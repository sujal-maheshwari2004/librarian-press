"""
server.py — background HTTP server exposing /metrics in Prometheus text format.

Pull model, exactly like Prometheus: the process exposes an endpoint and a
scraper (Prometheus, Grafana Alloy/Agent, etc.) pulls it on an interval. Runs on
a daemon thread so it never blocks training or the chat REPL.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import REGISTRY

_server = None


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/metrics", "/"):
            body = REGISTRY.render().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok\n")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # don't spam stdout on every scrape


def start_server(port: int, host: str = "0.0.0.0"):
    """Start the metrics server once. Idempotent; returns the server (or None on failure)."""
    global _server
    if _server is not None:
        return _server
    try:
        _server = ThreadingHTTPServer((host, port), _Handler)
    except OSError as e:
        print(f"[metrics] could not bind {host}:{port} ({e}); metrics disabled")
        return None
    thread = threading.Thread(target=_server.serve_forever, daemon=True)
    thread.start()
    print(f"[metrics] Prometheus metrics at http://{host}:{port}/metrics")
    return _server
