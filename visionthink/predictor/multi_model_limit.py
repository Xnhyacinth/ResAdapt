import os
import time
import threading
import asyncio
import queue
import itertools
import gc
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =========================
# Worker
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
    GPU worker:
      - load predictor once
      - single or micro-batch execution
      - on CUDA_OOM: split batch recursively until success / single item fails
      - non-blocking drain: get_nowait to fill microbatch quickly
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    torch.cuda.set_device(0)
    # Performance: enable cuDNN benchmarking and TF32 where available to improve throughput
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        # PyTorch exposes matmul TF32 flag on newer versions
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    from visionthink.predictor.modeling_predictor import PredictorForConditionalGeneration
    from visionthink.adaptive.utils import compute_scales_and_sample_means_cpu, _to_cpu_deep

    predictor = PredictorForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto", trust_remote_code=True
    )
    predictor.eval()

    max_scale = float(getattr(predictor.config, "max_scale", 2.0))
    min_scale = float(getattr(predictor.config, "min_scale", 0.25))
    use_discrete_action = bool(getattr(predictor.config, "use_discrete_action", False))
    discrete_step = float(getattr(predictor.config, "discrete_step", 0.25))

    _step = 0

    def _extract_echoes(payload: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
        try:
            return payload.get("_doc_id", None), payload.get("_compound_id", None)
        except Exception:
            return None, None

    def _put_result(req_id, ok: bool, payload: Dict[str, Any]):
        out_q.put((req_id, gpu_id, payload))

    def run_one(req_id: str, payload: Dict[str, Any]):
        nonlocal _step
        _step += 1
        step = _step

        doc_id_echo, compound_echo = _extract_echoes(payload)

        try:
            with torch.inference_mode():
                out = predictor(**payload)

            scales_cpu, scale_mask_cpu, sample_means_cpu = compute_scales_and_sample_means_cpu(
                out,
                max_scale=max_scale,
                min_scale=min_scale,
                use_discrete_action=use_discrete_action,
                default_step=discrete_step,
            )

            del out

            if step % 64 == 0:
                gc.collect()

            _put_result(req_id, True, {
                "ok": True,
                "scale_means": sample_means_cpu,
                "scale_mask": scale_mask_cpu,
                "scales": scales_cpu,
                "_doc_id": doc_id_echo,
                "_compound_id": compound_echo,
            })

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            print(f"[gpu{gpu_id}] CUDA_OOM step={step}: {e}", file=sys.stderr, flush=True)
            _put_result(req_id, False, {
                "ok": False,
                "err": "CUDA_OOM",
                "retry": False,
                "_doc_id": doc_id_echo,
                "_compound_id": compound_echo,
            })

        except Exception as e:
            torch.cuda.empty_cache()
            print(f"[gpu{gpu_id}] ERROR step={step}: {repr(e)}", file=sys.stderr, flush=True)
            _put_result(req_id, False, {
                "ok": False,
                "err": repr(e),
                "retry": False,
                "_doc_id": doc_id_echo,
                "_compound_id": compound_echo,
            })

    def _run_batch_once(sub_items: List[Tuple[str, Dict[str, Any]]]):
        # build batch_messages
        batch_messages = []
        eval_mode = True
        return_mm_data = False

        for _, payload in sub_items:
            msgs = payload.get("messages")
            if msgs is None:
                raise RuntimeError("payload missing 'messages'")

            # allow payload["messages"] == [messages]
            if isinstance(msgs, list) and msgs and isinstance(msgs[0], list) and len(msgs) == 1:
                msgs = msgs[0]

            batch_messages.append(msgs)
            eval_mode = payload.get("eval_mode", True)  # assume consistent in batch
            if payload.get("return_mm_data", False):
                return_mm_data = True

        with torch.inference_mode():
            out = predictor(messages=batch_messages, eval_mode=eval_mode, return_mm_data=return_mm_data)

        scales_cpu, scale_mask_cpu, sample_means_cpu = compute_scales_and_sample_means_cpu(
            out,
            max_scale=max_scale,
            min_scale=min_scale,
            use_discrete_action=use_discrete_action,
            default_step=discrete_step,
        )

        mm_list_cpu = None
        if return_mm_data:
            mm_list = out.get("multi_modal_data", None)
            if not isinstance(mm_list, list) or len(mm_list) != len(sub_items):
                raise RuntimeError(
                    f"Bad predictor multi_modal_data len={None if mm_list is None else len(mm_list)} expected={len(sub_items)}"
                )
            mm_list_cpu = [_to_cpu_deep(mm) for mm in mm_list]

        del out

        # emit
        for i, (req_id, payload) in enumerate(sub_items):
            doc_id_echo, compound_echo = _extract_echoes(payload)
            resp = {
                "ok": True,
                "scale_means": sample_means_cpu[i],
                "scale_mask": (scale_mask_cpu[i] if scale_mask_cpu is not None and hasattr(scale_mask_cpu, "__len__") else scale_mask_cpu),
                "scales": scales_cpu[i],
                "_doc_id": doc_id_echo,
                "_compound_id": compound_echo,
            }
            if return_mm_data:
                resp["scaled_multi_modal_data"] = [mm_list_cpu[i]]
            _put_result(req_id, True, resp)

    def run_batch(items: List[Tuple[str, Dict[str, Any]]]):
        nonlocal _step
        _step += 1
        step = _step

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
                    doc_id_echo, compound_echo = _extract_echoes(payload)
                    _put_result(req_id, False, {
                        "ok": False,
                        "err": "CUDA_OOM",
                        "retry": False,
                        "_doc_id": doc_id_echo,
                        "_compound_id": compound_echo,
                    })
                    continue

                mid = len(sub) // 2
                stack.append(sub[mid:])
                stack.append(sub[:mid])

            except Exception as e:
                torch.cuda.empty_cache()
                for req_id, payload in sub:
                    doc_id_echo, compound_echo = _extract_echoes(payload)
                    _put_result(req_id, False, {
                        "ok": False,
                        "err": repr(e),
                        "retry": False,
                        "_doc_id": doc_id_echo,
                        "_compound_id": compound_echo,
                    })

        if step % 64 == 0:
            gc.collect()

    # ---- main loop ----
    if not enable_batch:
        while True:
            item = in_q.get()
            if item is None:
                break
            req_id, payload = item
            run_one(req_id, payload)
        return

    # batch mode: time window + max size + drain nowait
    buf: List[Tuple[str, Dict[str, Any]]] = []
    poll_timeout = max(0.001, microbatch_ms / 1000.0)
    last_flush = time.time()

    while True:
        # 1) blocking get with timeout
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

        buf.append(item)

        while len(buf) < microbatch_max:
            try:
                nxt = in_q.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                # flush what we have then exit
                if buf:
                    run_batch(buf)
                return
            buf.append(nxt)

        if len(buf) >= microbatch_max or (time.time() - last_flush) >= poll_timeout:
            run_batch(buf)
            buf = []
            last_flush = time.time()


# =========================
# Client / Dispatcher
# =========================

@dataclass
class _PendingSync:
    done: threading.Event
    result: Optional[Dict[str, Any]] = None
    err: Optional[BaseException] = None
    gpu_id: int = -1

class MultiGPUInferPool:
    def __init__(
        self,
        model_path: str,
        num_gpus: int = 8,
        max_queue_per_gpu: int = 8,
        timeout: float = 600.0,
        max_total_inflight: Optional[int] = 32,
        max_inflight_per_gpu: int = 4,
        enable_batch: bool = True,
        microbatch_ms: int = 10,
        microbatch_max: int = 16,
        submit_threads: int = 64,
    ):
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.timeout = timeout
        self._shutdown = False
        self._started = False

        self._global_sem = threading.Semaphore(max_total_inflight) if max_total_inflight else None
        self._gpu_sems = [threading.Semaphore(max_inflight_per_gpu) for _ in range(num_gpus)]

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

        self._pending: Dict[int, _PendingSync] = {}
        self._pending_lock = threading.Lock()
        self._req_id_gen = itertools.count()

        self._inflight = [0] * num_gpus
        self._inflight_lock = threading.Lock()

        self._consumer_thread = None
        self._submit_executor = ThreadPoolExecutor(max_workers=submit_threads)

    def start(self):
        if self._started:
            return
        for p in self.procs:
            p.start()
        self._consumer_thread = threading.Thread(target=self._consume_results_forever, daemon=True)
        self._consumer_thread.start()
        self._started = True

    def close(self, timeout: float = 10.0):
        self._shutdown = True
        # stop workers
        for q in self.in_qs:
            try:
                q.put_nowait(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()

        self._submit_executor.shutdown(wait=False, cancel_futures=True)

        # fail all pending
        with self._pending_lock:
            for rid, pend in self._pending.items():
                pend.err = RuntimeError("Pool closed")
                pend.done.set()
            self._pending.clear()

    def _pick_gpu(self) -> int:
        # least inflight + token. Try fast path first, then a short blocking wait to avoid busy-spin.
        while True:
            with self._inflight_lock:
                order = sorted(range(self.num_gpus), key=lambda i: self._inflight[i])

            # Fast, non-blocking path: grab any available token from least-inflight GPUs.
            for g in order:
                if self._gpu_sems[g].acquire(blocking=False):
                    with self._inflight_lock:
                        self._inflight[g] += 1
                    return g

            # Slow path: wait briefly on semaphores in order to avoid CPU spinning
            for g in order:
                try:
                    if self._gpu_sems[g].acquire(timeout=0.05):
                        with self._inflight_lock:
                            self._inflight[g] += 1
                        return g
                except Exception:
                    continue

            # backoff
            time.sleep(0.01)

    def _release_gpu(self, g: int):
        with self._inflight_lock:
            self._inflight[g] = max(0, self._inflight[g] - 1)
        try:
            self._gpu_sems[g].release()
        except Exception:
            pass

    def submit_sync(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        if not self._started:
            raise RuntimeError("Pool not started")
        if self._shutdown:
            raise RuntimeError("Pool is shutting down")

        tmo = timeout or self.timeout

        if self._global_sem:
            if not self._global_sem.acquire(timeout=tmo):
                raise TimeoutError("global inflight timeout")

        req_id = next(self._req_id_gen)
        gpu = None
        try:
            gpu = self._pick_gpu()
            pend = _PendingSync(done=threading.Event(), gpu_id=gpu)
            with self._pending_lock:
                self._pending[req_id] = pend

            self.in_qs[gpu].put((req_id, payload), timeout=min(5.0, tmo))

            if not pend.done.wait(timeout=tmo):
                raise TimeoutError("request timeout")

            if pend.err is not None:
                raise pend.err
            return pend.result

        finally:
            if gpu is not None:
                with self._pending_lock:
                    still_pending = req_id in self._pending
                    if still_pending:
                        self._pending.pop(req_id, None)
                        self._release_gpu(gpu)
                        if self._global_sem:
                            self._global_sem.release()

    def submit_many_sync(self, payloads: List[Dict[str, Any]], timeout: Optional[float] = None):
        if not payloads:
            return []
        tmo = timeout or self.timeout
        futs = [self._submit_executor.submit(self.submit_sync, p, tmo) for p in payloads]
        outs = []
        for f in futs:
            try:
                outs.append(f.result(timeout=tmo + 5))
            except Exception as e:
                outs.append(e)
        return outs

    def _consume_results_forever(self):
        while not self._shutdown:
            try:
                req_id, gpu_id, result = self.out_q.get()
            except Exception:
                continue

            with self._pending_lock:
                pend = self._pending.pop(req_id, None)

            # release tokens
            self._release_gpu(gpu_id)
            if self._global_sem:
                try:
                    self._global_sem.release()
                except Exception:
                    pass

            if pend is None:
                continue
            pend.result = result
            pend.done.set()