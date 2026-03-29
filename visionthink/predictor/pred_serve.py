import ray
from ray import serve
from fastapi import FastAPI, Request, HTTPException
import torch

import os
import uuid
import asyncio
import time
from typing import Optional, Tuple, Dict, Set

from visionthink.adaptive.utils import decode_base64_to_numpy, serialize_tensor_to_base64

app = FastAPI()

FIXED_REPLICAS = 8 
MODEL_POLL_INTERVAL_SEC = 1.0
LOAD_TIMEOUT_SEC = 600


@ray.remote
class GlobalConfig:
    """
    A detached actor that stores the desired model path and an epoch counter.

    - epoch increments on each init (set_path).
    - ready_workers is tracked per epoch to avoid mixing signals across inits.
    - workers register themselves once on startup so we know current fleet.
    """
    def __init__(self):
        self.current_path: Optional[str] = None
        self.epoch: int = 0

        self.expected_replicas: int = FIXED_REPLICAS
        self.registered_workers: Set[str] = set()

        # epoch -> set(worker_id)
        self.ready_workers_by_epoch: Dict[int, Set[str]] = {}

    def register_worker(self, worker_id: str) -> int:
        self.registered_workers.add(worker_id)
        return len(self.registered_workers)

    def set_path(self, path: str, expected_replicas: int) -> Tuple[int, str, int]:
        self.epoch += 1
        self.current_path = path

        # Keep expected replicas consistent; if user passes something else, we clamp
        self.expected_replicas = int(expected_replicas)

        # init epoch ready set
        self.ready_workers_by_epoch[self.epoch] = set()
        return self.epoch, self.current_path, self.expected_replicas

    def get_state(self) -> Tuple[int, Optional[str], int, int]:
        """
        returns (epoch, current_path, expected_replicas, registered_workers_count)
        """
        return self.epoch, self.current_path, self.expected_replicas, len(self.registered_workers)

    def report_ready(self, worker_id: str, epoch: int, loaded_path: str) -> None:
        if epoch != self.epoch:
            # stale report
            return
        if loaded_path != self.current_path:
            return
        self.ready_workers_by_epoch.setdefault(epoch, set()).add(worker_id)

    def check_progress(self, epoch: int) -> Tuple[int, int, int]:
        """
        returns (ready_count, expected_replicas, registered_workers_count) for a given epoch
        """
        ready = len(self.ready_workers_by_epoch.get(epoch, set()))
        return ready, self.expected_replicas, len(self.registered_workers)


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": FIXED_REPLICAS, "max_replicas": FIXED_REPLICAS},
    max_ongoing_requests=5,
    max_queued_requests=128,
)
@serve.ingress(app)
class PredictorDeployment:
    def __init__(self):
        self.config_actor = ray.get_actor("GlobalConfigActor")

        self.worker_id = f"{os.uname().nodename}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.predictor = None
        self.loaded_path: Optional[str] = None
        self.loaded_epoch: int = -1

        self.load_lock = asyncio.Lock()
        self.loading_task: Optional[asyncio.Task] = None

        # Register worker in global config
        try:
            # fire and forget; we don't want __init__ to block too long
            ray.get(self.config_actor.register_worker.remote(self.worker_id))
        except Exception:
            pass

        # Start a watcher so every replica auto-loads after /init (no HTTP ping needed)
        self.watcher_task = asyncio.get_running_loop().create_task(self._watch_model_updates())

    async def _watch_model_updates(self):
        """
        Background loop: poll (epoch, path). If changed, trigger loading.
        This replaces the unreliable localhost ping/broadcast approach.
        """
        last_seen_epoch = -1
        while True:
            try:
                epoch, path, expected_replicas, _ = await self.config_actor.get_state.remote()
                if path is not None and epoch != last_seen_epoch:
                    last_seen_epoch = epoch
                    # Trigger async load in the background; don't block watcher
                    if self.loading_task is None or self.loading_task.done():
                        self.loading_task = asyncio.get_running_loop().create_task(
                            self._ensure_model_loaded(target_epoch=epoch, target_path=path)
                        )
            except Exception:
                # keep watcher alive
                pass

            await asyncio.sleep(MODEL_POLL_INTERVAL_SEC)

    async def _ensure_model_loaded(self, target_epoch: Optional[int] = None, target_path: Optional[str] = None):
        """
        Ensure current model matches GlobalConfig's latest (epoch, path).
        Uses epoch to avoid races across rapid re-inits.
        """
        # Fetch latest if not provided
        if target_epoch is None or target_path is None:
            target_epoch, target_path, _, _ = await self.config_actor.get_state.remote()

        if target_path is None:
            if self.loaded_path is None:
                raise HTTPException(status_code=503, detail="Not initialized")
            return

        # Fast path: already loaded for this epoch/path
        if self.loaded_epoch == target_epoch and self.loaded_path == target_path:
            await self.config_actor.report_ready.remote(self.worker_id, target_epoch, self.loaded_path)
            return

        async with self.load_lock:
            # Re-check inside lock
            latest_epoch, latest_path, _, _ = await self.config_actor.get_state.remote()
            if latest_path is None:
                if self.loaded_path is None:
                    raise HTTPException(status_code=503, detail="Not initialized")
                return

            # If init changed while waiting for lock, load the latest one
            target_epoch, target_path = latest_epoch, latest_path

            if self.loaded_epoch == target_epoch and self.loaded_path == target_path:
                await self.config_actor.report_ready.remote(self.worker_id, target_epoch, self.loaded_path)
                return

            print(f"[{self.worker_id}] Loading model epoch={target_epoch}, path={target_path}")

            # Cleanup old model
            if self.predictor is not None:
                try:
                    del self.predictor
                except Exception:
                    pass
                self.predictor = None
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Load model (blocking load in executor)
            def _load_sync(p: str):
                from visionthink.predictor.modeling_predictor import PredictorForConditionalGeneration
                return PredictorForConditionalGeneration.from_pretrained(
                    p, dtype="auto", device_map="auto", trust_remote_code=True
                )

            loop = asyncio.get_running_loop()
            predictor = await loop.run_in_executor(None, _load_sync, target_path)

            # Check again: if epoch changed during load, discard & let watcher re-trigger
            latest_epoch2, latest_path2, _, _ = await self.config_actor.get_state.remote()
            if latest_epoch2 != target_epoch or latest_path2 != target_path:
                print(f"[{self.worker_id}] ⚠️ Discarding stale load epoch={target_epoch} (latest={latest_epoch2})")
                try:
                    del predictor
                except Exception:
                    pass
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return

            self.predictor = predictor
            self.predictor.eval()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.loaded_epoch = target_epoch
            self.loaded_path = target_path
            print(f"[{self.worker_id}] ✅ Loaded successfully epoch={target_epoch}")

        await self.config_actor.report_ready.remote(self.worker_id, target_epoch, self.loaded_path)

    @app.post("/init")
    async def initialize_model(self, request: Request):
        data = await request.json()
        new_path = data.get("predictor_path")
        requested_replicas = int(data.get("num_replicas", FIXED_REPLICAS))

        if not new_path:
            raise HTTPException(status_code=400, detail="miss 'predictor_path'")

        # Hard align replicas with deployment config to avoid dead-wait / mismatch bugs
        if requested_replicas != FIXED_REPLICAS:
            # You can choose to 400 here; I clamp + return warning for robustness
            requested_replicas = FIXED_REPLICAS

        epoch, path, expected = await self.config_actor.set_path.remote(new_path, requested_replicas)

        # Immediately trigger local replica load (others will load via watcher)
        if self.loading_task is None or self.loading_task.done():
            self.loading_task = asyncio.get_running_loop().create_task(
                self._ensure_model_loaded(target_epoch=epoch, target_path=path)
            )

        return {
            "status": "accepted",
            "epoch": epoch,
            "message": f"Initialization started for {path}. Expected replicas={expected}. Replicas will auto-load via watcher."
        }

    @app.post("/predict")
    async def predict(self, request: Request):
        try:
            data = await request.json()
        except Exception as e:
            raw_body = await request.body()
            print(f"❌ JSON Parse Error! Raw body was: {raw_body}")
            print(f"❌ Error detail: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON or Client Disconnect")

        # Lightweight health check
        if data.get("check_health", False):
            latest_epoch, latest_path, _, _ = await self.config_actor.get_state.remote()
            if self.loaded_epoch == latest_epoch and self.loaded_path == latest_path and self.loaded_path is not None:
                return {"status": "ready", "worker": self.worker_id, "epoch": self.loaded_epoch}
            else:
                # ensure loading is scheduled
                if self.loading_task is None or self.loading_task.done():
                    self.loading_task = asyncio.get_running_loop().create_task(self._ensure_model_loaded())
                return {"status": "loading", "worker": self.worker_id, "loaded_epoch": self.loaded_epoch}

        # Ensure model loaded
        await self._ensure_model_loaded()

        def _inference_pipeline(input_data):
            messages = input_data.get("messages", [])
            multi_modal_data = input_data.get("multi_modal_data", None)
            
            if multi_modal_data is not None:
                multi_modal_data = [{
                    "images": [decode_base64_to_numpy(img) for img in mm["images"]] if "images" in mm else None,
                    "videos": [(decode_base64_to_numpy(vid[0]), vid[1]) for vid in mm["videos"]] if "videos" in mm else None
                } for mm in multi_modal_data]

            input_data["messages"] = messages if multi_modal_data is None else [
                (message, mm) for message, mm in zip(messages, multi_modal_data)
            ]
            
            output = self.predictor(**input_data)

            scaled_mm = output["multi_modal_data"]
            for mm in scaled_mm:
                if "images" in mm:
                    mm["images"] = [serialize_tensor_to_base64(img) for img in mm["images"]]
                if "videos" in mm:
                    mm["videos"] = [
                        (serialize_tensor_to_base64(vid[0]), vid[1]) 
                        for vid in mm["videos"]
                    ]
            return scaled_mm

        loop = asyncio.get_running_loop()
        try:
            scaled_multi_modal_data = await loop.run_in_executor(None, _inference_pipeline, data)

        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(status_code=507, detail="CUDA OOM")

        except RuntimeError as e:
            print(f"❌ Inference Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        # messages = data.get("messages", None)
        # if messages is None:
        #     raise HTTPException(status_code=400, detail="miss 'messages'")

        # multi_modal_data = data.get("multi_modal_data", None)
        # if multi_modal_data is not None:
        #     multi_modal_data = [{
        #         "images": [decode_base64_to_numpy(img) for img in mm["images"]] if "images" in mm else None,
        #         "videos": [(decode_base64_to_numpy(vid[0]), vid[1]) for vid in mm["videos"]] if "videos" in mm else None
        #     } for mm in multi_modal_data]

        # data["messages"] = messages if multi_modal_data is None else [
        #     (message, mm) for message, mm in zip(messages, multi_modal_data)
        # ]

        # loop = asyncio.get_running_loop()
        # try:
        #     output = await loop.run_in_executor(None, lambda: self.predictor(**data))

        # except torch.cuda.OutOfMemoryError:
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        #     raise HTTPException(status_code=507, detail="CUDA OOM")

        # Be defensive: output may not always include multi_modal_data
        # scaled_multi_modal_data = output.get("multi_modal_data", [])
        # for mm in scaled_multi_modal_data:
        #     if "images" in mm and mm["images"] is not None:
        #         mm["images"] = [serialize_tensor_to_base64(img) for img in mm["images"]]
        #     if "videos" in mm and mm["videos"] is not None:
        #         mm["videos"] = [(serialize_tensor_to_base64(vid[0]), vid[1]) for vid in mm["videos"]]

        return {"scaled_multi_modal_data": scaled_multi_modal_data}


if __name__ == "__main__":
    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except ConnectionError:
        ray.init(ignore_reinit_error=True)

    serve.start(http_options={"host": "0.0.0.0", "port": 8000})

    try:
        ray.get_actor("GlobalConfigActor")
    except ValueError:
        GlobalConfig.options(name="GlobalConfigActor", lifetime="detached").remote()

    serve.run(PredictorDeployment.bind(), route_prefix="/")
    print(f"🚀 {FIXED_REPLICAS}-GPU parallel inference service is up and running!")

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Stopping serve...")
