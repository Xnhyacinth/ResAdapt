import os
import uuid
import time
import threading
import asyncio
import queue
import itertools
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =========================
# Worker side
# =========================

def _gpu_worker_main(
    gpu_id: int,
    model_path: str,
    in_q: mp.Queue,
    out_q: mp.Queue,
    enable_batch: bool = False,
    microbatch_ms: int = 10,
    microbatch_max: int = 8,
):
    """
    One process per GPU. Loads model once.
    - If enable_batch: micro-batch collects requests then calls predictor once with batched messages.
    - Otherwise: process one-by-one.
    """

    # Best practice: set CUDA_VISIBLE_DEVICES before importing torch (avoid CUDA init surprises).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    torch.cuda.set_device(0)

    from visionthink.predictor.modeling_predictor import PredictorForConditionalGeneration

    predictor = PredictorForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto", trust_remote_code=True
    )
    predictor.eval()

    def run_one(req_id: str, payload: Dict[str, Any]):
        try:
            with torch.inference_mode():
                out = predictor(**payload)

            out_q.put((req_id, gpu_id, {"ok": True, "scaled_multi_modal_data": out["multi_modal_data"]}))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out_q.put((req_id, gpu_id, {"ok": False, "err": "CUDA_OOM", "retry": True}))
        except Exception as e:
            out_q.put((req_id, gpu_id, {"ok": False, "err": repr(e), "retry": False}))

    def run_batch(items: List[Tuple[str, Dict[str, Any]]]):
        """
        True batch path (optional):
        predictor supports messages=list batch.
        We assume payload structure contains at least "messages".
        We will batch by stacking payload["messages"].
        """
        try:
            batch_messages = []
            eval_mode = True

            for _, payload in items:
                msgs = payload.get("messages")

                # allow payload["messages"] == [messages]
                if isinstance(msgs, list) and msgs and isinstance(msgs[0], list) and len(msgs) == 1:
                    msgs = msgs[0]

                batch_messages.append(msgs)
                eval_mode = payload.get("eval_mode", True)

            with torch.inference_mode():
                out = predictor(messages=batch_messages, eval_mode=eval_mode)

            mm_list = out.get("multi_modal_data")
            if not isinstance(mm_list, list) or len(mm_list) != len(items):
                raise RuntimeError(f"Bad predictor output len={len(mm_list)} expected={len(items)}")

            for (req_id, _), mm in zip(items, mm_list):
                out_q.put((req_id, gpu_id, {"ok": True, "scaled_multi_modal_data": [mm]}))

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for req_id, _ in items:
                out_q.put((req_id, gpu_id, {"ok": False, "err": "CUDA_OOM", "retry": True}))
        except Exception as e:
            for req_id, _ in items:
                out_q.put((req_id, gpu_id, {"ok": False, "err": repr(e), "retry": False}))

    if not enable_batch:
        # simple mode
        while True:
            item = in_q.get()
            if item is None:
                break
            req_id, payload = item
            run_one(req_id, payload)
        return

    # batch mode: micro-batch window
    buf: List[Tuple[str, Dict[str, Any]]] = []
    poll_timeout = max(0.001, microbatch_ms / 1000.0)
    last_flush = time.time()

    while True:
        try:
            item = in_q.get(timeout=poll_timeout)
        except queue.Empty:
            if buf and (time.time() - last_flush) >= poll_timeout:
                run_batch(buf)
                buf = []
                last_flush = time.time()
            continue

        if item is None:
            if buf:
                run_batch(buf)
            break

        req_id, payload = item
        buf.append((req_id, payload))

        if len(buf) >= microbatch_max:
            run_batch(buf)
            buf = []
            last_flush = time.time()
            continue

        if (time.time() - last_flush) >= poll_timeout:
            run_batch(buf)
            buf = []
            last_flush = time.time()


# =========================
# Client / dispatcher side
# =========================

@dataclass
class _Pending:
    fut: asyncio.Future
    gpu_id: int
    created_at: float


class MultiGPUInferPool:
    """
    Optimized multiprocess pool:
    - One process per GPU, model loaded once per process.
    - Least-inflight scheduling
    - Global inflight limiting
    - Efficient result polling (blocking consumer thread)
    - Optional micro-batch in worker
    """

    def __init__(
        self,
        model_path: str,
        num_gpus: int = 8,
        max_queue_per_gpu: int = 8,
        timeout: float = 600.0,
        max_total_inflight: Optional[int] = 32,
        enable_batch: bool = True,
        microbatch_ms: int = 10,
        microbatch_max: int = 8,
        oom_max_retries: int = 0,
    ):
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.max_queue_per_gpu = max_queue_per_gpu
        self.timeout = timeout

        self.enable_batch = enable_batch
        self.microbatch_ms = microbatch_ms
        self.microbatch_max = microbatch_max
        self.oom_max_retries = oom_max_retries

        self._started = False
        self._shutdown = False

        self._global_sem = threading.Semaphore(max_total_inflight) if max_total_inflight else None

        ctx = mp.get_context("spawn")
        self.in_qs = [ctx.Queue(maxsize=max_queue_per_gpu) for _ in range(num_gpus)]
        self.out_q = ctx.Queue()

        self.procs = [
            ctx.Process(
                target=_gpu_worker_main,
                args=(i, model_path, self.in_qs[i], self.out_q, enable_batch, microbatch_ms, microbatch_max),
                daemon=True,
            )
            for i in range(num_gpus)
        ]

        self._pending: Dict[str, _Pending] = {}
        self._req_id_gen = itertools.count()
        self._inflight = [0] * num_gpus
        self._lock = threading.Lock()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        self._consumer_thread: Optional[threading.Thread] = None

        self._poll_future = None  # concurrent.futures.Future for consumer dispatch start

    def start(self):
        if self._started:
            return
        for p in self.procs:
            p.start()

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        # blocking consumer thread: out_q.get -> dispatch into asyncio loop
        self._consumer_thread = threading.Thread(target=self._consume_results_forever, daemon=True)
        self._consumer_thread.start()

        self._started = True

    def close(self, timeout: float = 10.0):
        self._shutdown = True

        # fail pending
        if self._loop:
            def _fail_all():
                for rid, p in list(self._pending.items()):
                    if not p.fut.done():
                        p.fut.set_exception(RuntimeError("Pool closed"))
                self._pending.clear()
            self._loop.call_soon_threadsafe(_fail_all)

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

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread:
            self._loop_thread.join(timeout=2)
        if self._consumer_thread:
            self._consumer_thread.join(timeout=2)

    # ---------- scheduling ----------

    def _pick_gpu_least_inflight(self) -> int:
        with self._lock:
            g = min(range(self.num_gpus), key=lambda i: self._inflight[i])
            self._inflight[g] += 1
            return g

    def _decr_inflight(self, gpu_id: int):
        with self._lock:
            self._inflight[gpu_id] = max(0, self._inflight[gpu_id] - 1)

    # ---------- submit API ----------

    async def submit(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        if not self._started:
            raise RuntimeError("Pool not started. Call start() first.")
        if self._shutdown:
            raise RuntimeError("Pool is shutting down")

        # global inflight limiter (recommended with vLLM sharing GPUs)
        if self._global_sem:
            await asyncio.get_running_loop().run_in_executor(None, self._global_sem.acquire)

        # req_id = uuid.uuid4().hex
        req_id = next(self._req_id_gen)
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        gpu = self._pick_gpu_least_inflight()
        self._pending[req_id] = _Pending(fut=fut, gpu_id=gpu, created_at=time.time())

        try:
            await loop.run_in_executor(None, self.in_qs[gpu].put, (req_id, payload))
            result = await asyncio.wait_for(fut, timeout=timeout or self.timeout)
            return result
        except Exception:
            # on timeout/exception: cleanup pending + inflight + sem
            self._pending.pop(req_id, None)
            self._decr_inflight(gpu)
            if self._global_sem:
                try:
                    self._global_sem.release()
                except Exception:
                    pass
            raise

    def submit_sync(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        if self._loop is None:
            raise RuntimeError("Pool not started. Call start() first.")
        cfut = asyncio.run_coroutine_threadsafe(self.submit(payload, timeout=timeout), self._loop)
        return cfut.result(timeout=(timeout or self.timeout) + 5)

    # ---------- result consumer ----------

    def _consume_results_forever(self):
        """
        Dedicated blocking consumer thread to avoid spawning executor tasks for each out_q.get.
        """
        while not self._shutdown:
            try:
                req_id, gpu_id, result = self.out_q.get()
            except Exception:
                continue

            if self._loop is None:
                continue

            # dispatch to event loop thread safely
            self._loop.call_soon_threadsafe(self._dispatch_result, req_id, gpu_id, result)

    def _dispatch_result(self, req_id: str, gpu_id: int, result: Dict[str, Any]):
        """
        Runs in event loop thread.
        """
        self._decr_inflight(gpu_id)

        # release global inflight slot
        if self._global_sem:
            try:
                self._global_sem.release()
            except Exception:
                pass

        pending = self._pending.pop(req_id, None)
        if not pending:
            return
        if pending.fut.done():
            return

        pending.fut.set_result(result)
