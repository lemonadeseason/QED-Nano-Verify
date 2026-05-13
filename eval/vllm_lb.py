"""Lightweight round-robin load balancer for multiple vLLM servers.

Usage:
    python vllm_lb.py  # starts on port 8080, forwards to 4 vLLM nodes

Then point the official eval at http://localhost:8080/v1
"""
import http.server
import urllib.request
import threading
import itertools
import sys

BACKENDS = [
    "http://100.64.198.94:8000",   # node-0
    "http://100.65.93.230:8000",   # node-1
    "http://100.64.162.169:8000",  # node-2
    "http://100.64.229.1:8000",    # node-3
]

LISTEN_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
_cycle = itertools.cycle(BACKENDS)
_lock = threading.Lock()


def next_backend():
    with _lock:
        return next(_cycle)


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def _proxy(self):
        backend = next_backend()
        url = backend + self.path
        body = None
        if 'Content-Length' in self.headers:
            body = self.rfile.read(int(self.headers['Content-Length']))

        req = urllib.request.Request(url, data=body, method=self.command)
        for key, val in self.headers.items():
            if key.lower() not in ('host', 'content-length'):
                req.add_header(key, val)
        if body:
            req.add_header('Content-Length', str(len(body)))

        try:
            with urllib.request.urlopen(req, timeout=18000) as resp:
                status = resp.status
                resp_headers = resp.headers
                resp_body = resp.read()
        except urllib.error.HTTPError as e:
            status = e.code
            resp_headers = e.headers
            resp_body = e.read()
        except Exception as e:
            self.send_error(502, str(e))
            return

        self.send_response(status)
        for key, val in resp_headers.items():
            if key.lower() not in ('transfer-encoding', 'connection'):
                self.send_header(key, val)
        self.end_headers()
        self.wfile.write(resp_body)

    def log_message(self, format, *args):
        pass  # suppress logs


if __name__ == '__main__':
    server = http.server.ThreadingHTTPServer(('0.0.0.0', LISTEN_PORT), ProxyHandler)
    print(f"Load balancer on :{LISTEN_PORT} -> {BACKENDS}")
    server.serve_forever()
