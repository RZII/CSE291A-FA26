"""HTTP-level checks for admission, FIFO order, streaming and safe fallback.

The fake runtime reports a large idle cache deliberately: observing retained
tokens must not deadlock the admission policy. No model or AWS is required.
"""

import concurrent.futures
import contextlib
import json
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("test condition timed out")
        time.sleep(0.005)


class RuntimeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.estimates = []
        self.generations = []
        self.ends = {}
        self.active = self.peak = 0
        self.malformed_slots = False


def runtime_handler(state):
    class RuntimeHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.respond(200, [{"n_prompt_tokens": (float("inf") if state.malformed_slots else 999999),
                                "is_processing": False}])

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            marker = body["marker"]
            if self.path.endswith("/input_tokens"):
                with state.lock:
                    state.estimates.append(marker)
                time.sleep(body.get("estimate_delay", 0))
                if body.get("estimate_incomplete"):
                    self.send_response(200)
                    self.send_header("Content-Length", "200")
                    self.end_headers()
                    self.wfile.write(b"{}")
                    return
                self.respond(503 if body.get("estimate_fail") else 200,
                             {"input_tokens": body.get("test_tokens", 60)})
                return
            with state.lock:
                state.generations.append((marker, body, time.monotonic()))
                state.active += 1
                state.peak = max(state.peak, state.active)
            try:
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(b'data: {"choices":[]}\n\n')
                    self.wfile.flush()
                    time.sleep(body.get("duration", 0.15))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    time.sleep(body.get("duration", 0.15))
                    self.respond(200, {"marker": marker})
            finally:
                with state.lock:
                    state.ends[marker] = time.monotonic()
                    state.active -= 1
    return RuntimeHandler


@contextlib.contextmanager
def scheduler(policy="kv-admission", queue_budget=5):
    state = RuntimeState()
    runtime = ThreadingHTTPServer(("127.0.0.1", 0), runtime_handler(state))
    runtime_thread = threading.Thread(target=runtime.serve_forever, daemon=True)
    runtime_thread.start()
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    with tempfile.TemporaryDirectory() as temporary:
        log = Path(temporary) / "scheduler.jsonl"
        command = [sys.executable, str(ROOT / "scheduler.py"), "--k", "3",
                   "--port", str(port), "--queue-capacity", "16", "--max-connections", "16",
                   "--model-url", f"http://127.0.0.1:{runtime.server_port}/v1/chat/completions",
                   "--policy", policy, "--kv-budget", "100", "--kv-headroom", "1",
                   "--kv-margin-tokens", "0", "--network-timeout", "3",
                   "--queue-budget", str(queue_budget), "--log", str(log)]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        def ready():
            if process.poll() is not None:
                raise AssertionError("scheduler exited before ready")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    return True
            except OSError:
                return False
        until(ready)

        def send(marker, **overrides):
            body = {"model": "fake-model", "messages": [{"role": "user", "content": marker}],
                    "marker": marker, "test_tokens": 60, "max_tokens": 20, "stream": False,
                    "tools": [{"type": "function", "function": {"name": "test"}}],
                    "chat_template_kwargs": {"enable_thinking": False}, **overrides}
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                    "X-Run-ID": "test", "X-Task-ID": marker, "X-Agent-ID": marker,
                    "X-Request-ID": marker, "X-Step-ID": "1"})
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    return response.status, response.read(), body
            except urllib.error.HTTPError as error:
                return error.code, error.read(), body

        try:
            yield state, send, log
        finally:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise AssertionError("scheduler did not drain and stop")
            runtime.shutdown()
            runtime.server_close()


class MemoryAdmissionTests(unittest.TestCase):
    def test_fifo_budget_and_unchanged_body_despite_estimator_reordering(self):
        with scheduler() as (state, send, log), concurrent.futures.ThreadPoolExecutor(3) as pool:
            futures = []
            for marker, delay in [("a", 0.09), ("b", 0.01), ("c", 0)]:
                futures.append(pool.submit(send, marker, estimate_delay=delay))
                until(lambda: marker in state.estimates)
            results = [future.result() for future in futures]
            self.assertEqual([row[0] for row in state.generations], ["a", "b", "c"])
            self.assertEqual(state.peak, 1)
            self.assertTrue(all(result[0] == 200 for result in results))
            self.assertEqual({row[0]: row[1] for row in state.generations},
                             {result[2]["marker"]: result[2] for result in results})
            until(lambda: log.exists() and len(log.read_text().splitlines()) == 3)
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertTrue(all(row["policy_observed_resident_tokens"] == 999999 for row in rows))
            self.assertTrue(all(row["policy_reservation_tokens"] == 80 for row in rows))
            self.assertGreater(max(row["policy_wait_ms"] for row in rows), 100)

    def test_small_requests_still_batch_under_budget(self):
        with scheduler() as (state, send, _), concurrent.futures.ThreadPoolExecutor(3) as pool:
            futures = [pool.submit(send, marker, test_tokens=10, max_tokens=10) for marker in "abc"]
            self.assertTrue(all(future.result()[0] == 200 for future in futures))
            self.assertEqual(state.peak, 3)

    def test_oversize_and_estimator_failure_run_alone_without_rejection(self):
        with scheduler() as (state, send, log), concurrent.futures.ThreadPoolExecutor(3) as pool:
            first = pool.submit(send, "oversize", test_tokens=200)
            until(lambda: "oversize" in state.estimates)
            second = pool.submit(send, "unknown", estimate_fail=True)
            until(lambda: "unknown" in state.estimates)
            third = pool.submit(send, "normal")
            self.assertEqual([first.result()[0], second.result()[0], third.result()[0]], [200] * 3)
            self.assertEqual(state.peak, 1)
            until(lambda: log.exists() and len(log.read_text().splitlines()) == 3)
            reasons = {row["request_id"]: row["policy_reason"] for row in
                       (json.loads(line) for line in log.read_text().splitlines())}
            self.assertEqual(reasons, {"oversize": "oversize_single", "unknown": "estimate_error_single",
                                       "normal": "within_budget"})

    def test_stream_holds_reservation_until_relay_finishes(self):
        with scheduler() as (state, send, _), concurrent.futures.ThreadPoolExecutor(2) as pool:
            first = pool.submit(send, "stream", stream=True, duration=0.2)
            until(lambda: any(row[0] == "stream" for row in state.generations))
            second = pool.submit(send, "next")
            first_result, second_result = first.result(), second.result()
            self.assertIn(b"[DONE]", first_result[1])
            self.assertEqual(second_result[0], 200)
            self.assertEqual(state.peak, 1)
            next_start = next(row[2] for row in state.generations if row[0] == "next")
            self.assertGreaterEqual(next_start, state.ends["stream"])

    def test_expired_ticket_does_not_block_following_requests(self):
        with scheduler(queue_budget=0.05) as (state, send, _), concurrent.futures.ThreadPoolExecutor(2) as pool:
            first = pool.submit(send, "active", duration=0.2)
            until(lambda: any(row[0] == "active" for row in state.generations))
            second = pool.submit(send, "expires")
            self.assertEqual(second.result()[0], 504)
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(send("after")[0], 200)

    def test_default_fifo_keeps_fixed_concurrency_and_skips_estimation(self):
        with scheduler(policy="fifo") as (state, send, _), concurrent.futures.ThreadPoolExecutor(3) as pool:
            futures = [pool.submit(send, marker) for marker in "abc"]
            self.assertTrue(all(future.result()[0] == 200 for future in futures))
            self.assertEqual(state.peak, 3)
            self.assertEqual(state.estimates, [])

    def test_malformed_native_response_falls_back_without_stranding_fifo_ticket(self):
        with scheduler() as (state, send, log):
            state.malformed_slots = True
            self.assertEqual(send("truncated", estimate_incomplete=True)[0], 200)
            self.assertEqual(send("following")[0], 200)
            until(lambda: log.exists() and len(log.read_text().splitlines()) == 2)
            rows = {row["request_id"]: row for row in
                    (json.loads(line) for line in log.read_text().splitlines())}
            self.assertEqual(rows["truncated"]["policy_reason"], "estimate_error_single")
            self.assertIn("IncompleteRead", rows["truncated"]["policy_estimate_error"])
            self.assertIn("OverflowError", rows["following"]["policy_observation_error"])


if __name__ == "__main__":
    unittest.main()
