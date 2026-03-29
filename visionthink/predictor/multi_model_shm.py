import os
import uuid
import time
import queue
import tempfile
import threading
import asyncio
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =========================
# Helpers: shm IO + to_cpu
# =========================

def _shm_dir() -> Optional[str]:
    return "/dev/shm" if os.path.exists("/dev/shm") else None


def _to_cpu_deep(x: Any) -> Any:
    """Recursively move torch.Tensors to CPU; keep metadata unchanged."""
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().to("cpu")
    except Exception:
        pass

    if isinstance(x, dict):
        return {k: _to_cpu_deep(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cpu_deep(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_cpu_deep(v) for v in x)
    return x


def _dump_obj_to_shm(obj: Any, prefix: str = "mm_") -> str:
    """torch.save obj into /dev/shm (or /tmp) and return file path."""
    import torch
    d = _shm_dir()
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".pt", dir=d)
    os.close(fd)
    torch.save(obj, path)
    return path


def _safe_unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception:
        logger.warning(f"Failed to remove {path}", exc_info=True)


def cleanup_stale_shm_files(prefix: str = "mm_pool_", max_age_sec: int = 24 * 3600) -> int:
    """
    Best-effort cleanup for leftover shm files from previous crashes.
    Only removes files in /dev/shm matching prefix and older than max_age_sec.
    """
    d = _shm_dir()
    if not d:
        return 0
    now = time.time()
    removed = 0
    try:
        for name in os.listdir(d):
            if not name.startswith(prefix):
                continue
            path = os.path.join(d, name)
            try:
                st = os.stat(path)
                if now - st.st_mtime > max_age_sec:
                    os.remove(path)
                    removed += 1
            except Exception:
                continue
    except Exception:
        return 0
    return removed


# =========================
# Worker side (per GPU proc)
# =========================

def _gpu_worker_main(
    gpu_id: int,
    model_path: str,
    in_q: mp.Queue,
    out_q: mp.Queue,
    microbatch_ms: int,
    microbatch_max: int,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    torch.cuda.set_device(0)

    from visionthink.predictor.modeling_predictor import PredictorForConditionalGeneration

    predictor = PredictorForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto", trust_remote_code=True
    )
    predictor.eval()

    def run_batch(batch_items):
        """
        batch_items: List[(req_id, payload)]
        payload must contain: {"messages": [one_request_messages], "eval_mode": True, ...}
        predictor supports batching by concatenating messages:
          predictor(messages=[m1, m2, ...], eval_mode=True)
        """
        try:
            # 1) build batch messages
            batch_messages = []
            for _, payload in batch_items:
                # each payload["messages"] is already a list (one request)
                # we append that one request messages as an element of the batch list
                # NOTE: your earlier code used {"messages": [messages]} (extra nesting)
                # Here we standardize to payload["messages"] = messages (one request)
                batch_messages.append(payload["messages"])

            # 2) call predictor once
            with torch.no_grad():
                out = predictor(messages=batch_messages, eval_mode=True)

            mm_list = out["multi_modal_data"]
            if not isinstance(mm_list, list) or len(mm_list) != len(batch_items):
                raise RuntimeError(f"Bad predictor output: len(multi_modal_data)={len(mm_list)} vs batch={len(batch_items)}")

            # 3) write each element to shm and respond
            for (req_id, _), mm in zip(batch_items, mm_list):
                mm_cpu = _to_cpu_deep(mm)  # move tensors to cpu
                mm_path = _dump_obj_to_shm(mm_cpu, prefix=f"mm_pool_gpu{gpu_id}_")
                out_q.put((req_id, gpu_id, {"ok": True, "mm_path": mm_path}))

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # mark all retryable
            for req_id, _ in batch_items:
                out_q.put((req_id, gpu_id, {"ok": False, "err": "CUDA_OOM", "retry": True}))
        except Exception as e:
            for req_id, _ in batch_items:
                out_q.put((req_id, gpu_id, {"ok": False, "err": repr(e), "retry": False}))

    # micro-batch buffer
    buf = []
    poll_timeout = max(0.001, microbatch_ms / 1000.0)
    last_flush = time.time()

    while True:
        try:
            item = in_q.get(timeout=poll_timeout)
        except queue.Empty:
            # time-based flush
            if buf and (time.time() - last_flush) >= poll_timeout:
                run_batch(buf)
                buf = []
                last_flush = time.time()
            continue
        except EOFError:
            break

        if item is None:
            if buf:
                run_batch(buf)
            break

        req_id, payload = item

        # ✅ normalize payload shape: require payload["messages"] is one request's messages
        # If caller sends {"messages": [messages]}, unwrap it here
        if isinstance(payload.get("messages"), list) and payload["messages"] and isinstance(payload["messages"][0], list):
            # payload["messages"] = [messages] -> messages
            if len(payload["messages"]) == 1:
                payload = dict(payload)
                payload["messages"] = payload["messages"][0]

        buf.append((req_id, payload))

        # size-based flush
        if len(buf) >= microbatch_max:
            run_batch(buf)
            buf = []
            last_flush = time.time()
            continue

        # time-based flush
        if (time.time() - last_flush) >= poll_timeout:
            run_batch(buf)
            buf = []
            last_flush = time.time()



# =========================
# Dispatcher side
# =========================

@dataclass
class _Pending:
    fut: asyncio.Future
    payload: Dict[str, Any]
    gpu_id: int
    created_at: float = field(default_factory=time.time)
    retries: int = 0
    max_retries: int = 2
    mm_path: Optional[str] = None  # track for cleanup when set


class MultiGPUInferPool:
    """
    - One worker process per GPU (each loads model once).
    - Returns mm_path to shm file containing CPU multi_modal_data.
    - inflight-aware scheduling (least inflight).
    - optional micro-batch at worker side.
    """

    def __init__(
        self,
        model_path: str,
        num_gpus: int = 8,
        max_queue_per_gpu: int = 8,
        timeout: float = 600.0,
        max_total_inflight: Optional[int] = None,
        microbatch_ms: int = 10,
        microbatch_max: int = 4,
        cleanup_prefix: str = "mm_pool_",
        cleanup_on_start: bool = True,
    ):
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.max_queue_per_gpu = max_queue_per_gpu
        self.timeout = timeout

        # global inflight limiter (recommended when sharing GPUs with vLLM)
        self.max_total_inflight = max_total_inflight
        self._global_sem = threading.Semaphore(max_total_inflight) if max_total_inflight else None

        # micro-batch config (worker local)
        self.microbatch_ms = microbatch_ms
        self.microbatch_max = microbatch_max

        # shm cleanup
        self.cleanup_prefix = cleanup_prefix
        self.cleanup_on_start = cleanup_on_start

        ctx = mp.get_context("spawn")
        self.in_qs = [ctx.Queue(maxsize=max_queue_per_gpu) for _ in range(num_gpus)]
        self.out_q = ctx.Queue()

        self.procs = [
            ctx.Process(
                target=_gpu_worker_main,
                args=(i, model_path, self.in_qs[i], self.out_q, microbatch_ms, microbatch_max),
                daemon=True,
            )
            for i in range(num_gpus)
        ]

        # state
        self._pending: Dict[str, _Pending] = {}
        self._inflight: List[int] = [0] * num_gpus
        self._lock = threading.Lock()
        self._shutdown = False

        # loop thread
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        # result consumer thread (blocking out_q.get)
        self._consumer_thread: Optional[threading.Thread] = None

    def start(self):
        if self.cleanup_on_start:
            removed = cleanup_stale_shm_files(prefix="mm_pool_", max_age_sec=24 * 3600)
            if removed:
                logger.info(f"[MultiGPUInferPool] cleaned {removed} stale shm files")

        for p in self.procs:
            p.start()

        # background asyncio loop for awaiting submit futures
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        # blocking consumer thread for out_q.get -> dispatch into asyncio loop
        self._consumer_thread = threading.Thread(target=self._consume_results_forever, daemon=True)
        self._consumer_thread.start()

    def close(self, timeout: float = 10.0):
        """
        Best-effort shutdown.
        """
        self._shutdown = True

        # cancel all pending futures and cleanup their shm
        if self._loop:
            def _cancel_all():
                for rid, p in list(self._pending.items()):
                    if not p.fut.done():
                        p.fut.set_exception(RuntimeError("Pool closed"))
                    # cleanup path if already produced
                    _safe_unlink(p.mm_path)
                self._pending.clear()
            self._loop.call_soon_threadsafe(_cancel_all)

        # stop workers
        for q in self.in_qs:
            try:
                q.put(None, block=False)
            except Exception:
                pass

        for p in self.procs:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()

        # stop loop
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=2)
        if self._consumer_thread:
            self._consumer_thread.join(timeout=2)

    # ---------- scheduling ----------

    def _pick_gpu_least_inflight(self) -> int:
        with self._lock:
            # choose least inflight; tie-break by index
            g = min(range(self.num_gpus), key=lambda i: self._inflight[i])
            self._inflight[g] += 1
            return g

    def _decr_inflight(self, gpu_id: int):
        with self._lock:
            self._inflight[gpu_id] = max(0, self._inflight[gpu_id] - 1)

    # ---------- submit API ----------

    async def submit(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Returns:
          {"ok": True, "scaled_multi_modal_data": <CPU multi_modal_data>}
          or {"ok": False, "err": ..., "retry": bool}
        """
        if self._shutdown:
            raise RuntimeError("Pool is shutting down")
        if self._loop is None:
            raise RuntimeError("Pool not started")

        # optional global inflight limit
        if self._global_sem:
            # acquire in a thread to avoid blocking event loop
            await asyncio.get_running_loop().run_in_executor(None, self._global_sem.acquire)

        req_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        gpu = self._pick_gpu_least_inflight()

        pending = _Pending(fut=fut, payload=payload, gpu_id=gpu)
        self._pending[req_id] = pending

        try:
            await loop.run_in_executor(None, self.in_qs[gpu].put, (req_id, payload))
            result = await asyncio.wait_for(fut, timeout=timeout or self.timeout)
            return result
        finally:
            # if submit() exits early due to exception/timeout, ensure sem released
            # (normal path releases sem in dispatch when result set; this is fallback)
            # We guard with try to avoid double-release, handled in dispatch.
            pass

    def submit_sync(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        if self._loop is None:
            raise RuntimeError("Pool not started")
        cfut = asyncio.run_coroutine_threadsafe(self.submit(payload, timeout=timeout), self._loop)
        return cfut.result(timeout=(timeout or self.timeout) + 5)

    # ---------- result handling ----------

    def _consume_results_forever(self):
        """
        Dedicated blocking consumer thread:
        out_q.get() -> dispatch to asyncio loop
        """
        while not self._shutdown:
            try:
                req_id, gpu_id, result = self.out_q.get()
            except Exception:
                continue
            if self._loop is None:
                continue
            self._loop.call_soon_threadsafe(self._dispatch_result, req_id, gpu_id, result)

    def _dispatch_result(self, req_id: str, gpu_id: int, result: Dict[str, Any]):
        """
        Runs in pool asyncio loop thread (thread-safe).
        """
        pending = self._pending.get(req_id)
        self._decr_inflight(gpu_id)

        # release global sem if any (one request completed)
        if self._global_sem:
            try:
                self._global_sem.release()
            except Exception:
                pass

        if not pending:
            # unknown request: cleanup shm if present
            _safe_unlink(result.get("mm_path"))
            return

        # If request already done/timed out, cleanup and drop
        if pending.fut.done():
            _safe_unlink(result.get("mm_path"))
            self._pending.pop(req_id, None)
            return

        # retry on OOM: resubmit to another gpu (or same least inflight)
        if (not result.get("ok")) and result.get("retry", False) and pending.retries < pending.max_retries:
            pending.retries += 1
            # resubmit with same req_id? simpler: create new req_id and chain future
            # We'll create a new req_id and move the pending record.
            new_req_id = uuid.uuid4().hex
            new_gpu = self._pick_gpu_least_inflight()
            pending.gpu_id = new_gpu
            self._pending[new_req_id] = pending
            self._pending.pop(req_id, None)

            # schedule put in executor to avoid blocking loop
            async def _resubmit():
                await asyncio.get_running_loop().run_in_executor(None, self.in_qs[new_gpu].put, (new_req_id, pending.payload))
            asyncio.create_task(_resubmit())
            return

        # ok path: load from shm and cleanup
        if result.get("ok"):
            mm_path = result.get("mm_path")
            pending.mm_path = mm_path
            try:
                import torch
                scaled_multi_modal_data = torch.load(mm_path, map_location="cpu")
                out = {"ok": True, "scaled_multi_modal_data": scaled_multi_modal_data}
                pending.fut.set_result(out)
            except Exception as e:
                pending.fut.set_result({"ok": False, "err": f"LOAD_MM_FAILED: {repr(e)}", "retry": False})
            finally:
                _safe_unlink(mm_path)
                pending.mm_path = None
                self._pending.pop(req_id, None)
            return

        # error path
        pending.fut.set_result(result)
        self._pending.pop(req_id, None)
