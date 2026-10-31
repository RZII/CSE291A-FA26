"""
精简记录版：FIFO + K，保留有界队列/连接、错误返回和正常停止。
日志每次请求只写一行，不再记录一长串阶段事件和重复的原始时间。
所有代码在本文件；完整响应仍原样交给 Agent。
"""
import argparse
import json
import queue
import signal
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

# 配置：执行数、等待容量、连接数是三个不同限制。默认值同原版，均可用命令行覆盖：
#   python scheduler.py --k 4 --queue-capacity 256 --log logs/run_x/scheduler.jsonl
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--k", type=int, default=2, help="固定并发 K")
argument_parser.add_argument("--queue-capacity", type=int, default=8, help="等待队列容量，满了返回 429")
argument_parser.add_argument("--max-connections", type=int, default=16, help="接收连接上限，满了返回 503")
argument_parser.add_argument("--model-url", default="http://127.0.0.1:8080/v1/chat/completions")
argument_parser.add_argument("--port", type=int, default=8000)
argument_parser.add_argument("--network-timeout", type=float, default=60, help="秒；流式下是两块数据之间最多等多久")
argument_parser.add_argument("--queue-budget", type=float, default=300, help="秒；排队超过即返回 504")
argument_parser.add_argument("--log", default="work/requests_minimal.jsonl")
argument_parser.add_argument("--policy", choices=["fifo", "kv-admission"], default="fifo")
argument_parser.add_argument("--kv-budget", type=int, default=0, help="共享 KV 池总 token 数；kv-admission 必填")
argument_parser.add_argument("--kv-margin-tokens", type=int, default=256, help="每个在途请求额外预留的 token 数")
argument_parser.add_argument("--kv-headroom", type=float, default=0.90, help="准入最多预留总 KV 预算的这一比例")
arguments = argument_parser.parse_args()
if arguments.k < 1:
    argument_parser.error("--k must be positive")
if arguments.policy == "kv-admission" and arguments.kv_budget < 1:
    argument_parser.error("kv-admission requires positive --kv-budget")
if arguments.kv_margin_tokens < 0 or not 0 < arguments.kv_headroom <= 1:
    argument_parser.error("--kv-margin-tokens must be nonnegative and --kv-headroom must be in (0, 1]")

concurrency = arguments.k
queue_capacity = arguments.queue_capacity
maximum_connections = arguments.max_connections
maximum_body_bytes = 1024 * 1024
model_url = arguments.model_url
server_address = ("127.0.0.1", arguments.port)
network_timeout_seconds = arguments.network_timeout  # 流式下是“两块数据之间最多等多久”，不是整段回答的总时长
queue_wait_budget_seconds = arguments.queue_budget  # baseline 里相当于不丢请求：排队多久本身就是要测的量
log_path = Path(arguments.log)
model_origin = urllib.parse.urlunsplit((*urllib.parse.urlsplit(model_url)[:2], "", "", ""))

waiting_queue = queue.Queue(maxsize=queue_capacity)
connection_slots = threading.BoundedSemaphore(maximum_connections)
admission_lock = threading.Lock()
stopping = threading.Event()


def runtime_json(path, body=None):
    """Native llama.cpp read-only/token-count endpoints; never change generation body."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(model_origin + path, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=min(2.0, network_timeout_seconds)) as response:
        return json.load(response)


def estimate_reservation(request_record):
    """Estimate active demand, not allocator free space or an eviction guarantee.

    This pinned runtime's input_tokens endpoint uses the same chat-template and
    tools parser as generation. Resident idle slot tokens are telemetry only:
    waiting for those to disappear would deadlock because the runtime clears or
    reuses them only when another request arrives.
    """
    started = time.monotonic()
    record = request_record
    body = record["body"]
    reservation = None
    try:
        count = runtime_json("/v1/chat/completions/input_tokens", body)["input_tokens"]
        output = body.get("max_tokens", body.get("max_completion_tokens"))
        if type(count) is not int or count < 0:
            raise ValueError("invalid_input_tokens")
        record["policy_prompt_tokens"] = count
        if type(output) is not int or output < 0:
            raise ValueError("unbounded_or_missing_output_limit")
        reservation = count + output + arguments.kv_margin_tokens
        record["policy_output_reservation_tokens"] = output
    except Exception as error:
        # A truncated HTTP body can raise http.client.IncompleteRead rather than
        # OSError; any native counter failure must use the logged solo fallback.
        record["policy_estimate_error"] = type(error).__name__ + ": " + str(error)
    try:
        slots = runtime_json("/slots")
        record["policy_observed_resident_tokens"] = sum(int(s.get("n_prompt_tokens", 0)) for s in slots)
        record["policy_observed_active_slots"] = sum(bool(s.get("is_processing")) for s in slots)
    except Exception as error:
        record["policy_observation_error"] = type(error).__name__ + ": " + str(error)
    record["policy_estimate_ms"] = (time.monotonic() - started) * 1000
    record["policy_reservation_tokens"] = reservation
    return reservation


class KVAdmission:
    """FIFO tickets + active token reservations, bounded by the original K.

    Unknown demand and a request larger than the soft budget run alone, without
    dropping the task or changing its output limit. Such requests can still
    overflow the real runtime. Cache occupancy and fragmentation remain outside
    this estimator's control and must be measured separately.
    """
    def __init__(self, budget, maximum_active):
        self.budget = budget
        self.maximum_active = maximum_active
        self.condition = threading.Condition()
        self.issued = 0
        self.next_ticket = 0
        self.cancelled = set()
        self.active = {}

    def issue(self):
        with self.condition:
            ticket = self.issued
            self.issued += 1
            return ticket

    def advance(self):
        while self.next_ticket in self.cancelled:
            self.cancelled.remove(self.next_ticket)
            self.next_ticket += 1

    def cancel(self, ticket):
        with self.condition:
            if ticket < self.next_ticket:
                return
            self.cancelled.add(ticket)
            self.advance()
            self.condition.notify_all()

    def admit(self, record):
        reservation = estimate_reservation(record)
        started = time.monotonic()
        ticket = record["policy_ticket"]
        deadline = record["enqueued_time"] + queue_wait_budget_seconds
        with self.condition:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.cancelled.add(ticket)
                    self.advance()
                    self.condition.notify_all()
                    record["policy_wait_ms"] = (time.monotonic() - started) * 1000
                    record["policy_reason"] = "queue_expired"
                    return False
                known_active = all(value is not None for value in self.active.values())
                fits = (reservation is not None and known_active and
                        sum(self.active.values()) + reservation <= self.budget)
                if ticket == self.next_ticket and len(self.active) < self.maximum_active and (not self.active or fits):
                    if reservation is None:
                        reason = "estimate_error_single"
                    elif reservation > self.budget:
                        reason = "oversize_single"
                    else:
                        reason = "within_budget"
                    self.active[ticket] = reservation
                    self.next_ticket += 1
                    self.advance()
                    record.update(policy_reason=reason, policy_budget_tokens=self.budget,
                                  policy_active_requests_at_admit=len(self.active),
                                  policy_active_tokens_at_admit=(sum(self.active.values()) if known_active and reservation is not None else None),
                                  policy_wait_ms=(time.monotonic() - started) * 1000)
                    self.condition.notify_all()
                    return True
                self.condition.wait(timeout=min(remaining, 0.5))

    def release(self, ticket):
        with self.condition:
            self.active.pop(ticket, None)
            # Also cancel an unadmitted ticket if an unexpected worker error
            # occurred; otherwise all later FIFO tickets could remain blocked.
            if ticket >= self.next_ticket:
                self.cancelled.add(ticket)
                self.advance()
            self.condition.notify_all()


kv_admission = (KVAdmission(max(1, int(arguments.kv_budget * arguments.kv_headroom)), concurrency)
                if arguments.policy == "kv-admission" else None)

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
        "step": request_record["step"],
        "k": concurrency,
        "model_started_unix_s": request_record["model_started_unix_s"],
        "queue_ms": request_record["queue_ms"],
        "model_request_ms": request_record["model_request_ms"],
        "total_ms": total_ms,
        "http_status": request_record["status"],
        "response_sent": response_sent,
        "error": request_record["error"],
        "policy": arguments.policy,
    }
    log_record.update({key: value for key, value in request_record.items() if key.startswith("policy_")})
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


def stream_model(request_record):
    """流式：打开到模型的连接，交给接收线程一边收一边转发给 Agent。

    整段回答转发完之前本函数不返回，所以这个工作人员一直占着 K 个名额之一。
    timeout 对每一次读取单独计时，所以流式下它的含义是“两块数据之间最多等多久”。
    """
    request_text = json.dumps(request_record["body"], ensure_ascii=False)
    headers = {"Content-Type": "application/json"}
    backend_request = urllib.request.Request(model_url, data=request_text.encode("utf-8"), headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(backend_request, timeout=network_timeout_seconds)
    except urllib.error.HTTPError as error:
        # 模型还没开始生成就报错（例如上下文太长）：按普通 JSON 错误原样返回。
        request_record["status"] = error.code
        request_record["response"] = error.read()
        request_record["error"] = "backend_http_error"
        error.close()
        return
    request_record["status"] = response.status
    request_record["stream"] = response
    request_record["notice"].set()  # 通知接收线程：可以开始转发了
    request_record["relay_done"].wait()  # 等接收线程转发完
    response.close()


def scheduler_worker(worker_number):
    while True:
        request_record = waiting_queue.get()
        if request_record is None:
            waiting_queue.task_done()
            return
        reserved = False
        try:
            if kv_admission is not None:
                reserved = kv_admission.admit(request_record)
            # ==================== 计时：排队 开始 ====================
            # enqueued_time 在接收线程里填好，取出后相减即可，不保存更多时间点。
            with admission_lock:
                queue_ms = (time.monotonic() - request_record["enqueued_time"]) * 1000
                request_record["queue_ms"] = queue_ms
            # ==================== 计时：排队 结束 ====================

            if (kv_admission is not None and not reserved) or queue_ms > queue_wait_budget_seconds * 1000:
                request_record["status"] = 504
                request_record["error"] = "queue_expired"
                request_record["response"] = error_response("queue_expired")
            else:
                # ==================== 附加：模型调用计时 开始 ====================
                model_started_time = time.monotonic()
                request_record["model_started_time"] = model_started_time
                request_record["model_started_unix_s"] = time.time()  # 与 Agent 事件日志对齐用
                # ==================== 附加：模型调用计时 结束 ====================

                try:
                    if request_record["body"].get("stream") is True:
                        stream_model(request_record)
                    else:
                        status_code, response_bytes = call_model(request_record)
                        request_record["status"] = status_code
                        request_record["response"] = response_bytes
                        if not 200 <= status_code < 300:
                            request_record["error"] = "backend_http_error"
                finally:
                    # ==================== 附加：模型调用耗时 开始 ====================
                    # 流式请求的耗时由接收线程在转发结束、写日志之前记录，这里不覆盖。
                    if "stream" not in request_record:
                        request_record["model_request_ms"] = (time.monotonic() - model_started_time) * 1000
                    # ==================== 附加：模型调用耗时 结束 ====================

        except Exception as error:
            request_record["status"] = 502
            request_record["error"] = str(error)
            request_record["response"] = error_response(str(error))
        finally:
            if kv_admission is not None:
                kv_admission.release(request_record["policy_ticket"])
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
            "relay_done": threading.Event(),  # 流式：接收线程转发完后通知工作人员
            "status": None,
            "response": None,
            "error": None,

            # ==================== 附加：身份与最少计时字段 开始 ====================
            "run_id": self.headers.get("X-Run-ID", ""),
            "task_id": self.headers.get("X-Task-ID", ""),
            "agent_id": self.headers.get("X-Agent-ID", ""),
            "request_id": self.headers.get("X-Request-ID", ""),
            "step": self.headers.get("X-Step-ID", ""),
            "timestamp_unix_s": time.time(),
            "model_started_unix_s": None,
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
            # 保留原接口五个标签的要求；step 也写进本层日志，便于按步统计排队。
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
            if not isinstance(request_body.get("stream", False), bool):
                raise ValueError("stream must be true or false")
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
                    if kv_admission is not None:
                        request_record["policy_ticket"] = kv_admission.issue()
                    waiting_queue.put_nowait(request_record)
                except queue.Full:
                    if kv_admission is not None:
                        kv_admission.cancel(request_record["policy_ticket"])
                    rejection = "queue_full"
        if rejection is not None:
            request_record["status"] = 429 if rejection == "queue_full" else 503
            request_record["error"] = rejection
            request_record["response"] = error_response(rejection)
            self.send_result(request_record)
            return

        finished_notice = request_record["notice"]
        finished_notice.wait()
        if "stream" in request_record:
            self.send_stream(request_record)
        else:
            self.send_result(request_record)

    def send_stream(self, request_record):
        """把模型的流式回答转给 Agent：每收到一行就立刻写出去。"""
        response_sent = False
        self.connection.settimeout(network_timeout_seconds)
        try:
            self.send_response(request_record["status"])
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Request-ID", request_record["request_id"])
            self.end_headers()  # 不写 Content-Length：转发完就关闭连接，Agent 读到连接关闭即结束
            for line in request_record["stream"]:
                self.wfile.write(line)
                self.wfile.flush()
            response_sent = True
        except OSError as error:
            # Agent 断开，或模型超过 network_timeout_seconds 没有新数据（TimeoutError 也是 OSError）
            request_record["error"] = "stream_interrupted: " + str(error)
        finally:
            # ==================== 附加：流式模型耗时 开始 ====================
            request_record["model_request_ms"] = (time.monotonic() - request_record["model_started_time"]) * 1000
            # ==================== 附加：流式模型耗时 结束 ====================
            request_record["relay_done"].set()  # 放开工作人员，它才能去取下一个请求
            # ==================== 附加：一个请求只写一行日志 开始 ====================
            write_log(request_record, response_sent)
            # ==================== 附加：一个请求只写一行日志 结束 ====================

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
print("准入策略：", arguments.policy,
      "KV soft budget=" + str(kv_admission.budget) if kv_admission else "", flush=True)
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
