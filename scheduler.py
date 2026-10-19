"""
精简记录版：FIFO + K，保留有界队列/连接、错误返回和正常停止。
日志每次请求只写一行，不再记录一长串阶段事件和重复的原始时间。
所有代码在本文件；完整响应仍原样交给 Agent。
"""
import json
import queue
import signal
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

# 配置：执行数、等待容量、连接数是三个不同限制。
concurrency = 2
queue_capacity = 8
maximum_connections = 16
maximum_body_bytes = 1024 * 1024
model_url = "http://127.0.0.1:8080/v1/chat/completions"#待对齐
server_address = ("127.0.0.1", 8000)
network_timeout_seconds = 20
queue_wait_budget_seconds = 10
log_path = Path("work/requests_minimal.jsonl")

waiting_queue = queue.Queue(maxsize=queue_capacity)
connection_slots = threading.BoundedSemaphore(maximum_connections)
admission_lock = threading.Lock()
stopping = threading.Event()

# ==================== 附加：日志准备 开始 ====================
log_lock = threading.Lock()
log_path.parent.mkdir(parents=True, exist_ok=True)
# ==================== 附加：日志准备 结束 ====================


def error_response(message):
    error_dictionary = {"error": message}
    error_text = json.dumps(error_dictionary, ensure_ascii=False)
    return error_text.encode("utf-8")


# ==================== 附加：日志函数 开始 ====================
def write_log(request_record, response_sent):
    total_ms = (time.monotonic() - request_record["received_time"]) * 1000
    log_record = {
        "timestamp_unix_s": request_record["timestamp_unix_s"],
        "run_id": request_record["run_id"],
        "task_id": request_record["task_id"],
        "agent_id": request_record["agent_id"],
        "request_id": request_record["request_id"],
        "queue_ms": request_record["queue_ms"],
        "model_request_ms": request_record["model_request_ms"],
        "total_ms": total_ms,
        "http_status": request_record["status"],
        "response_sent": response_sent,
        "error": request_record["error"],
    }
    log_text = json.dumps(log_record, ensure_ascii=False)
    try:
        with log_lock:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(log_text + "\n")
    except OSError as error:
        print("日志写入失败，本次实验数据不完整：", error, flush=True)
# ==================== 附加：日志函数 结束 ====================


def call_model(request_record):
    request_body = request_record["body"]
    request_text = json.dumps(request_body, ensure_ascii=False)
    request_bytes = request_text.encode("utf-8")

    headers = {"Content-Type": "application/json"}
    backend_request = urllib.request.Request(model_url, data=request_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(backend_request, timeout=network_timeout_seconds) as response:
            status_code = response.status
            response_bytes = response.read()
    except urllib.error.HTTPError as error:
        status_code = error.code
        response_bytes = error.read()
        error.close()
    return status_code, response_bytes


def scheduler_worker(worker_number):
    while True:
        request_record = waiting_queue.get()
        if request_record is None:
            waiting_queue.task_done()
            return
        try:
            # ==================== 计时：排队 开始 ====================
            # enqueued_time 在接收线程里填好，取出后相减即可，不保存更多时间点。
            with admission_lock:
                queue_ms = (time.monotonic() - request_record["enqueued_time"]) * 1000
                request_record["queue_ms"] = queue_ms
            # ==================== 计时：排队 结束 ====================

            if queue_ms > queue_wait_budget_seconds * 1000:
                request_record["status"] = 504
                request_record["error"] = "queue_expired"
                request_record["response"] = error_response("queue_expired")
            else:
                # ==================== 附加：模型调用计时 开始 ====================
                model_started_time = time.monotonic()
                # ==================== 附加：模型调用计时 结束 ====================

                try:
                    status_code, response_bytes = call_model(request_record)
                    request_record["status"] = status_code
                    request_record["response"] = response_bytes
                    if not 200 <= status_code < 300:
                        request_record["error"] = "backend_http_error"
                finally:
                    # ==================== 附加：模型调用耗时 开始 ====================
                    request_record["model_request_ms"] = (time.monotonic() - model_started_time) * 1000
                    # ==================== 附加：模型调用耗时 结束 ====================

        except Exception as error:
            request_record["status"] = 502
            request_record["error"] = str(error)
            request_record["response"] = error_response(str(error))
        finally:
            finished_notice = request_record["notice"]
            finished_notice.set()
            waiting_queue.task_done()


class AgentRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # request_record 仍是一本请求记录；身份和计时字段全部平铺。
        # body 是模型输入字典，notice 是线程通知，不能当作日志写进去。
        request_record = {
            "body": None,
            "notice": threading.Event(),
            "status": None,
            "response": None,
            "error": None,

            # ==================== 附加：身份与最少计时字段 开始 ====================
            "run_id": self.headers.get("X-Run-ID", ""),
            "task_id": self.headers.get("X-Task-ID", ""),
            "agent_id": self.headers.get("X-Agent-ID", ""),
            "request_id": self.headers.get("X-Request-ID", ""),
            "timestamp_unix_s": time.time(),
            "received_time": time.monotonic(),
            "enqueued_time": None,
            "queue_ms": None,
            "model_request_ms": None,
            # ==================== 附加：身份与最少计时字段 结束 ====================
        }
        self.connection.settimeout(network_timeout_seconds)
        try:
            if self.path != "/v1/chat/completions":
                raise ValueError("unknown path")
            # 保留原接口五个标签的要求，但 step 不再重复写进本层日志。
            for field in ["run_id", "task_id", "agent_id", "request_id"]:
                if not request_record[field].strip():
                    raise ValueError("missing identity: " + field)
            if int(self.headers.get("X-Step-ID", "0")) < 1:
                raise ValueError("X-Step-ID must be positive")
            body_length = int(self.headers.get("Content-Length", "0"))
            if body_length <= 0 or body_length > maximum_body_bytes:
                raise ValueError("invalid body size")
            request_bytes = self.rfile.read(body_length)
            if len(request_bytes) != body_length:
                raise ValueError("incomplete body")
            request_body = json.loads(request_bytes.decode("utf-8"))
            if not isinstance(request_body, dict):
                raise ValueError("body must be object")
            if not isinstance(request_body.get("messages"), list) or not request_body["messages"]:
                raise ValueError("messages must be nonempty list")
            if not isinstance(request_body.get("model"), str) or not request_body["model"]:
                raise ValueError("model must be nonempty text")
            if request_body.get("stream") is not False:
                raise ValueError("requires stream=false")
        except (ValueError, TypeError, UnicodeError, OSError) as error:
            request_record["status"] = 408 if isinstance(error, TimeoutError) else 400
            request_record["error"] = str(error)
            request_record["response"] = error_response(str(error))
            self.send_result(request_record)
            return

        request_record["body"] = request_body
        with admission_lock:
            if stopping.is_set():
                rejection = "server_stopping"
            else:
                rejection = None

                # ==================== 附加：入队前计时 开始 ====================
                request_record["enqueued_time"] = time.monotonic()
                # ==================== 附加：入队前计时 结束 ====================

                try:
                    waiting_queue.put_nowait(request_record)
                except queue.Full:
                    rejection = "queue_full"
        if rejection is not None:
            request_record["status"] = 429 if rejection == "queue_full" else 503
            request_record["error"] = rejection
            request_record["response"] = error_response(rejection)
            self.send_result(request_record)
            return

        finished_notice = request_record["notice"]
        finished_notice.wait()
        self.send_result(request_record)

    def send_result(self, request_record):
        response_sent = False
        self.connection.settimeout(network_timeout_seconds)
        try:
            self.send_response(request_record["status"])
            self.send_header("Content-Type", "application/json; charset=utf-8")
            response_bytes = request_record["response"]
            self.send_header("Content-Length", str(len(response_bytes)))
            self.send_header("X-Request-ID", request_record["request_id"])
            self.end_headers()
            self.wfile.write(response_bytes)
            self.wfile.flush()
            response_sent = True
        except OSError as error:
            request_record["error"] = "client_disconnected: " + str(error)
        finally:
            # ==================== 附加：一个请求只写一行日志 开始 ====================
            write_log(request_record, response_sent)
            # ==================== 附加：一个请求只写一行日志 结束 ====================


class BoundedHTTPServer(ThreadingHTTPServer):
    # 保留连接保护，不再为连接保护单独生成复杂事件字典。
    def process_request(self, connection, client_address):
        acquired = connection_slots.acquire(blocking=False)
        if not acquired:
            print("连接已满，返回503；未读取身份，由客户端记录本次失败。", flush=True)
            try:
                connection.settimeout(0.2)
                response = error_response("connection_limit")
                headers = "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: " + str(len(response)) + "\r\n\r\n"
                connection.sendall(headers.encode("ascii") + response)
            except OSError:
                pass
            self.shutdown_request(connection)
            return
        connection.settimeout(network_timeout_seconds)
        try:
            super().process_request(connection, client_address)
        except Exception:
            connection_slots.release()
            raise

    def process_request_thread(self, connection, client_address):
        try:
            super().process_request_thread(connection, client_address)
        finally:
            connection_slots.release()


def stop_signal(signal_number, frame):
    raise KeyboardInterrupt


http_server = BoundedHTTPServer(server_address, AgentRequestHandler)
http_server.daemon_threads = False
workers = []
for worker_index in range(concurrency):
    worker_number = worker_index + 1
    worker_thread = threading.Thread(target=scheduler_worker, args=(worker_number,))
    workers.append(worker_thread)
    worker_thread.start()
signal.signal(signal.SIGTERM, stop_signal)
print("精简记录版启动：", server_address, "K=", concurrency, flush=True)
print("日志：", log_path.resolve(), flush=True)
try:
    http_server.serve_forever()
except KeyboardInterrupt:
    print("停止接收，正在处理已入队请求。", flush=True)
finally:
    with admission_lock:
        stopping.set()
    http_server.socket.close()
    waiting_queue.join()
    for worker in workers:
        waiting_queue.put(None)
    for worker in workers:
        worker.join()
    http_server.server_close()
    print("已排空并停止。", flush=True)
