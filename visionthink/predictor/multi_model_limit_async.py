import os
import sys
import time
import gc
import queue
import itertools
import threading
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Iterable

import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, Future


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =============================================================================
# Worker
# =============================================================================

def _gpu_worker_main(
    gpu_id: int,
    model_path: str,
    in_q: mp.Queue,
    out_q: mp.Queue,
    *,
    enable_batch: bool,
    microbatch_ms: int,
    microbatch_max: int,
    max_frames: Optional[int] = None,
    max_scale_override: Optional[float] = None,
    min_scale_override: Optional[float] = None,
    send_ready: bool = True,
):
    """
    GPU worker process.

    - Uses CUDA_VISIBLE_DEVICES=gpu_id (1 GPU per proc).
    - Loads predictor once.
    - Supports micro-batching + recursive split on OOM.
    - Always returns result via out_q: (req_id, gpu_id, payload_dict)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    torch.cuda.set_device(0)

    # Perf toggles (safe)
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    try:
        # from visionthink.predictor.modeling_predictor import PredictorForConditionalGeneration
        from  transformers import AutoModel, AutoConfig
        from visionthink.adaptive.utils import compute_scales_and_sample_means_cpu, _to_cpu_deep
    except Exception as e:
        print(f"[worker gpu={gpu_id}] FATAL import error: {repr(e)}", file=sys.stderr, flush=True)
        return

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if max_frames is not None:
            try:
                max_frames_val = int(max_frames)
                if max_frames_val > 0:
                    config.max_frames = max_frames_val
            except Exception:
                pass
        if max_scale_override is not None:
            try:
                config.max_scale = float(max_scale_override)
            except Exception:
                pass
        if min_scale_override is not None:
            try:
                config.min_scale = float(min_scale_override)
            except Exception:
                pass
        predictor = AutoModel.from_pretrained(
            model_path, config=config, dtype="auto", device_map="auto", trust_remote_code=True
        )
        predictor.eval()
    except Exception as e:
        print(f"[worker gpu={gpu_id}] FATAL model load error: {repr(e)}", file=sys.stderr, flush=True)
        return

    max_scale = float(getattr(predictor.config, "max_scale", 2.0))
    min_scale = float(getattr(predictor.config, "min_scale", 0.25))
    if max_scale_override is not None:
        try:
            max_scale = float(max_scale_override)
        except Exception:
            pass
    if min_scale_override is not None:
        try:
            min_scale = float(min_scale_override)
        except Exception:
            pass
    use_discrete_action = bool(getattr(predictor.config, "use_discrete_action", False))
    discrete_step = float(getattr(predictor.config, "discrete_step", 0.25))

    def _extract_echoes(payload: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
        try:
            return payload.get("_doc_id", None), payload.get("_compound_id", None)
        except Exception:
            return None, None

    def _emit(req_id: int, ok: bool, resp: Dict[str, Any], payload: Optional[Dict[str, Any]] = None):
        """Always include echoes to help caller validate mapping."""
        if payload is not None:
            doc_id, compound_id = _extract_echoes(payload)
            resp["_doc_id"] = doc_id
            resp["_compound_id"] = compound_id
        resp["ok"] = bool(ok)
        out_q.put((req_id, gpu_id, resp))

    def _normalize_messages(msgs):
        # allow payload["messages"] == [messages]
        if isinstance(msgs, list) and msgs and isinstance(msgs[0], list) and len(msgs) == 1:
            return msgs[0]
        return msgs

    def _run_batch_once(items: List[Tuple[int, Dict[str, Any]]]):
        batch_messages = []
        eval_mode = True
        return_mm = False

        for _, payload in items:
            msgs = payload.get("messages", None)
            if msgs is None:
                raise RuntimeError("payload missing 'messages'")
            msgs = _normalize_messages(msgs)

            batch_messages.append(msgs)
            eval_mode = payload.get("eval_mode", True)
            if payload.get("return_mm_data", False):
                return_mm = True

        with torch.inference_mode():
            out = predictor(messages=batch_messages, eval_mode=eval_mode, return_mm_data=return_mm)

        scales_cpu, scale_mask_cpu, sample_means_cpu = compute_scales_and_sample_means_cpu(
            out,
            max_scale=max_scale,
            min_scale=min_scale,
            use_discrete_action=use_discrete_action,
            default_step=discrete_step,
        )

        mm_cpu = None
        if return_mm:
            mm_list = out.get("multi_modal_data", None)
            if not isinstance(mm_list, list) or len(mm_list) != len(items):
                raise RuntimeError(f"Bad predictor multi_modal_data len={None if mm_list is None else len(mm_list)} expected={len(items)}")
            mm_cpu = [_to_cpu_deep(m) for m in mm_list]

        del out

        for i, (req_id, payload) in enumerate(items):
            resp = {
                "scale_means": sample_means_cpu[i],
                "scale_mask": (scale_mask_cpu[i] if hasattr(scale_mask_cpu, "__len__") else scale_mask_cpu),
                "scales": scales_cpu[i],
            }
            if return_mm and mm_cpu is not None:
                resp["scaled_multi_modal_data"] = [mm_cpu[i]]
            _emit(req_id, True, resp, payload=payload)

    def _run_batch_with_oom_split(items: List[Tuple[int, Dict[str, Any]]]):
        # iterative split on OOM
        stack = [items]
        while stack:
            sub = stack.pop()
            if not sub:
                continue
            try:
                _run_batch_once(sub)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(sub) == 1:
                    req_id, payload = sub[0]
                    _emit(req_id, False, {"err": "CUDA_OOM", "retry": False}, payload=payload)
                    continue
                mid = len(sub) // 2
                stack.append(sub[mid:])
                stack.append(sub[:mid])
            except Exception as e:
                torch.cuda.empty_cache()
                for req_id, payload in sub:
                    _emit(req_id, False, {"err": repr(e), "retry": False}, payload=payload)

    # READY handshake (optional)
    if send_ready:
        try:
            out_q.put((-1, gpu_id, {"ok": True, "type": "READY"}))
        except Exception:
            pass

    # ---- main loop ----
    poll_timeout = max(0.001, microbatch_ms / 1000.0)
    log_every = int(os.getenv("MICRO_BATCH_LOG_EVERY", "0"))
    buf: List[Tuple[int, Dict[str, Any]]] = []
    last_flush = time.time()
    steps = 0

    while True:
        # blocking get with timeout
        try:
            item = in_q.get(timeout=poll_timeout)
        except queue.Empty:
            if enable_batch and buf and (time.time() - last_flush) >= poll_timeout:
                _run_batch_with_oom_split(buf)
                if log_every > 0 and steps % log_every == 0:
                    print(f"[worker gpu={gpu_id}] microbatch_size={len(buf)} microbatch_max={microbatch_max} microbatch_ms={microbatch_ms}")
                buf = []
                last_flush = time.time()
                steps += 1
                if steps % 64 == 0:
                    gc.collect()
            continue

        if item is None:
            # shutdown sentinel
            if buf:
                _run_batch_with_oom_split(buf)
            break

        req_id, payload = item

        if not enable_batch:
            _run_batch_with_oom_split([(req_id, payload)])
            if log_every > 0 and steps % log_every == 0:
                print(f"[worker gpu={gpu_id}] microbatch_size=1 microbatch_max={microbatch_max} microbatch_ms={microbatch_ms}")
            steps += 1
            if steps % 64 == 0:
                gc.collect()
            continue

        # batch mode
        buf.append((req_id, payload))

        # drain quickly without blocking
        while len(buf) < microbatch_max:
            try:
                nxt = in_q.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                if buf:
                    _run_batch_with_oom_split(buf)
                return
            buf.append(nxt)

        # flush conditions
        if len(buf) >= microbatch_max or (time.time() - last_flush) >= poll_timeout:
            _run_batch_with_oom_split(buf)
            buf = []
            last_flush = time.time()
            steps += 1
            if steps % 64 == 0:
                gc.collect()


# =============================================================================
# Client / Dispatcher
# =============================================================================

@dataclass
class _PendingSync:
    done: threading.Event
    gpu_id: int
    created_at: float
    result: Optional[Dict[str, Any]] = None
    err: Optional[BaseException] = None


class MultiGPUInferPool:
    """
    Robust multi-process GPU inference pool.

    Design goals:
      - never hang indefinitely: all blocking ops have timeouts
      - stable under high concurrency
      - explicit backpressure: global inflight + per-GPU inflight + per-GPU queue maxsize
      - microbatching inside each worker
    """

    def __init__(
        self,
        model_path: str,
        *,
        num_gpus: int = 8,
        max_queue_per_gpu: int = 8,
        timeout: float = 600.0,
        max_total_inflight: Optional[int] = 32,
        max_inflight_per_gpu: int = 4,
        enable_batch: bool = True,
        microbatch_ms: int = 10,
        microbatch_max: int = 32,
        max_frames: Optional[int] = None,
        max_scale: Optional[float] = None,
        min_scale: Optional[float] = None,
        schedule_policy: str = "least_inflight",
        submit_threads: int = 128,
        start_stagger_s: float = 0.0,
        send_ready: bool = True,
        start_ready_timeout_s: float = 120.0,
    ):
        self.model_path = model_path
        self.num_gpus = int(num_gpus)
        self.max_frames = max_frames
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.max_queue_per_gpu = int(max_queue_per_gpu)
        self.timeout = float(timeout)
        self.schedule_policy = str(schedule_policy or "least_inflight")

        self.enable_batch = bool(enable_batch)
        self.microbatch_ms = int(microbatch_ms)
        self.microbatch_max = int(microbatch_max)

        self.max_inflight_per_gpu = int(max_inflight_per_gpu)
        if self.max_inflight_per_gpu <= 0:
            raise ValueError("max_inflight_per_gpu must be >= 1")

        self._shutdown = False
        self._started = False

        # inflight limits
        self._global_sem = threading.Semaphore(max_total_inflight) if max_total_inflight else None
        self._gpu_sems = [threading.Semaphore(self.max_inflight_per_gpu) for _ in range(self.num_gpus)]

        # mp objects
        ctx = mp.get_context("spawn")
        self.in_qs = [ctx.Queue(maxsize=self.max_queue_per_gpu) for _ in range(self.num_gpus)]
        self.out_q = ctx.Queue()

        self.procs = [
            ctx.Process(
                target=_gpu_worker_main,
                args=(
                    i,
                    self.model_path,
                    self.in_qs[i],
                    self.out_q,
                ),
                kwargs=dict(
                    enable_batch=self.enable_batch,
                    microbatch_ms=self.microbatch_ms,
                    microbatch_max=self.microbatch_max,
                    max_frames=self.max_frames,
                    max_scale_override=self.max_scale,
                    min_scale_override=self.min_scale,
                    send_ready=send_ready,
                ),
                daemon=True,
            )
            for i in range(self.num_gpus)
        ]

        # pending map
        self._pending: Dict[int, _PendingSync] = {}
        self._pending_lock = threading.Lock()
        self._req_id_gen = itertools.count()

        # inflight counters (for least-inflight scheduling)
        self._inflight = [0] * self.num_gpus
        self._inflight_lock = threading.Lock()
        self._rr_cursor = 0
        self._rr_lock = threading.Lock()

        # threads
        self._consumer_thread: Optional[threading.Thread] = None
        self._submit_executor = ThreadPoolExecutor(max_workers=int(submit_threads))

        # start options
        self._start_stagger_s = float(start_stagger_s)
        self._send_ready = bool(send_ready)
        self._start_ready_timeout_s = float(start_ready_timeout_s)

        # ready tracking
        self._ready_mask = [False] * self.num_gpus
        self._ready_lock = threading.Lock()

    def start(self):
        if self._started:
            return

        logger.info(f"[Pool] starting {self.num_gpus} workers (spawn)...")
        for i, p in enumerate(self.procs):
            p.start()
            logger.info(f"[Pool] worker {i} started pid={p.pid}")
            if self._start_stagger_s > 0 and i < self.num_gpus - 1:
                time.sleep(self._start_stagger_s)

        # consumer thread first (so READY can be consumed)
        self._consumer_thread = threading.Thread(target=self._consume_results_forever, daemon=True)
        self._consumer_thread.start()

        # health check
        time.sleep(0.5)
        dead = [i for i, p in enumerate(self.procs) if not p.is_alive()]
        if dead:
            raise RuntimeError(f"[Pool] some workers died during startup: {dead}")

        # wait READY (optional)
        if self._send_ready:
            deadline = time.time() + self._start_ready_timeout_s
            while time.time() < deadline:
                with self._ready_lock:
                    if all(self._ready_mask):
                        break
                time.sleep(0.05)
            with self._ready_lock:
                ready_cnt = sum(1 for x in self._ready_mask if x)
            if ready_cnt != self.num_gpus:
                raise RuntimeError(f"[Pool] READY timeout: {ready_cnt}/{self.num_gpus} workers ready")

        self._started = True
        logger.info("[Pool] started.")

    def close(self, timeout: float = 10.0):
        self._shutdown = True

        # stop workers
        for q in self.in_qs:
            try:
                q.put_nowait(None)
            except Exception:
                pass

        for i, p in enumerate(self.procs):
            p.join(timeout=timeout)
            if p.is_alive():
                logger.warning(f"[Pool] worker {i} still alive -> terminate")
                p.terminate()

        # stop submit executor
        self._submit_executor.shutdown(wait=False, cancel_futures=True)

        # fail all pending
        with self._pending_lock:
            for rid, pend in self._pending.items():
                pend.err = RuntimeError("Pool closed")
                pend.done.set()
            self._pending.clear()

        logger.info("[Pool] closed.")

    def _pick_gpu(self, wait_timeout_s: float = 0.05, max_wait_s: float = 300.0) -> int:
        """
        Pick least-inflight GPU among those with available per-gpu token.
        Avoid tight spin: try nonblocking first, then small timed waits.
        
        Args:
            wait_timeout_s: Timeout for each semaphore acquire attempt.
            max_wait_s: Maximum total time to wait before raising TimeoutError.
            
        Raises:
            TimeoutError: If no GPU becomes available within max_wait_s seconds.
        """
        start_time = time.time()
        
        while True:
            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > max_wait_s:
                dead_workers = self._check_workers_healthy()
                raise TimeoutError(
                    f"_pick_gpu timeout after {elapsed:.1f}s. "
                    f"Dead workers: {dead_workers if dead_workers else 'none'}"
                )
            
            with self._inflight_lock:
                inflight_snapshot = list(self._inflight)
            if self.schedule_policy == "round_robin":
                with self._rr_lock:
                    start = self._rr_cursor
                    self._rr_cursor = (self._rr_cursor + 1) % self.num_gpus
                order = list(range(start, self.num_gpus)) + list(range(0, start))
            else:
                order = sorted(range(self.num_gpus), key=lambda i: (inflight_snapshot[i], i))
                with self._rr_lock:
                    if self.num_gpus > 1:
                        offset = self._rr_cursor % self.num_gpus
                        self._rr_cursor = (self._rr_cursor + 1) % self.num_gpus
                        order = order[offset:] + order[:offset]

            # fast path
            for g in order:
                if self._gpu_sems[g].acquire(blocking=False):
                    with self._inflight_lock:
                        self._inflight[g] += 1
                    return g

            # slow path
            for g in order:
                try:
                    if self._gpu_sems[g].acquire(timeout=wait_timeout_s):
                        with self._inflight_lock:
                            self._inflight[g] += 1
                        return g
                except Exception:
                    continue

            time.sleep(0.01)

    def _check_workers_healthy(self) -> List[int]:
        """Check if all worker processes are alive.
        
        Returns:
            List of GPU IDs whose workers are dead.
        """
        dead = [i for i, p in enumerate(self.procs) if not p.is_alive()]
        if dead:
            logger.warning(f"[Pool] Dead workers detected on GPUs: {dead}")
        return dead

    def _release_gpu(self, g: int):
        with self._inflight_lock:
            self._inflight[g] = max(0, self._inflight[g] - 1)
        try:
            self._gpu_sems[g].release()
        except Exception:
            pass

    def submit_sync(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Submit a single request and wait synchronously for result.
        
        Thread-safe. Applies backpressure via global + per-GPU semaphores.
        
        Args:
            payload: Request payload dict with 'messages' key required.
            timeout: Optional timeout in seconds, defaults to self.timeout.
            
        Returns:
            Result dict from worker.
            
        Raises:
            RuntimeError: If pool not started or shutting down.
            TimeoutError: If request times out.
        """
        if not self._started:
            raise RuntimeError("Pool not started")
        if self._shutdown:
            raise RuntimeError("Pool is shutting down")

        tmo = float(timeout or self.timeout)

        # Acquire global inflight token
        if self._global_sem:
            if not self._global_sem.acquire(timeout=tmo):
                raise TimeoutError("global inflight acquire timeout")

        req_id = next(self._req_id_gen)
        gpu = None
        tokens_released_by_consumer = False

        try:
            gpu = self._pick_gpu()

            pend = _PendingSync(done=threading.Event(), gpu_id=gpu, created_at=time.time())
            with self._pending_lock:
                self._pending[req_id] = pend

            # IMPORTANT: put must have timeout to avoid indefinite hang
            self.in_qs[gpu].put((req_id, payload), timeout=min(5.0, tmo))

            # wait result
            if not pend.done.wait(timeout=tmo):
                raise TimeoutError(f"request {req_id} timeout (no response after {tmo}s)")

            if pend.err is not None:
                raise pend.err

            # Success: consumer thread already released tokens and popped from pending
            tokens_released_by_consumer = True
            return pend.result  # type: ignore

        finally:
            # Only release tokens if consumer didn't handle it (timeout/exception case).
            # The consumer releases tokens BEFORE popping from _pending, so if req_id
            # is still in _pending, it means the consumer never processed this request.
            if gpu is not None and not tokens_released_by_consumer:
                with self._pending_lock:
                    if req_id in self._pending:
                        self._pending.pop(req_id, None)
                        self._release_gpu(gpu)
                        if self._global_sem:
                            try:
                                self._global_sem.release()
                            except Exception:
                                pass

    def submit_many_sync(self, payloads: List[Dict[str, Any]], timeout: Optional[float] = None) -> List[Any]:
        """
        High throughput batch submission:
          - concurrency bounded by submit_threads executor
          - each task uses submit_sync (safe timeouts)
        """
        if not payloads:
            return []

        from tqdm import tqdm
        from concurrent.futures import as_completed

        tmo = float(timeout or self.timeout)
        futures: List[Future] = [self._submit_executor.submit(self.submit_sync, p, tmo) for p in payloads]
        
        # Map futures to original index to preserve order
        future_to_idx = {f: i for i, f in enumerate(futures)}
        results = [None] * len(payloads)

        with tqdm(total=len(payloads), desc="Processing scales", unit="req") as pbar:
             for f in as_completed(futures, timeout=tmo + 10.0):
                 idx = future_to_idx[f]
                 try:
                     results[idx] = f.result()
                 except Exception as e:
                     results[idx] = e
                 pbar.update(1)
        
        return results

    def _consume_results_forever(self):
        """
        Receive (req_id, gpu_id, result) from out_q and wake up waiters.
        Must always release gpu/global semaphores to avoid deadlock.
        """
        while not self._shutdown:
            try:
                req_id, gpu_id, result = self.out_q.get()
            except Exception:
                continue

            # READY handshake
            if req_id == -1 and isinstance(result, dict) and result.get("type") == "READY":
                with self._ready_lock:
                    if 0 <= gpu_id < self.num_gpus:
                        self._ready_mask[gpu_id] = True
                continue

            # release tokens first to unblock schedulers
            self._release_gpu(gpu_id)
            if self._global_sem:
                try:
                    self._global_sem.release()
                except Exception:
                    pass

            # complete pending
            with self._pending_lock:
                pend = self._pending.pop(req_id, None)

            if pend is None:
                continue

            pend.result = result
            pend.done.set()
