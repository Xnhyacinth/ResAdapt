import argparse
import asyncio
import io
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

DIRECT_SYSTEM_PROMPT = "You are a helpful assistant."

COT_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your answer between the <answer> </answer> tags. "
    "The final answer MUST BE wrapped in \\boxed{} and the \\boxed{} expression MUST BE contained entirely within the <answer> </answer> tags."
)

SCALE_PROMPT = (
    "You are a cost-aware visual perception policy for long videos/images.\n"
    "Given a **video (sequence of frames)** or a **single image**, and a **question**, your job is to choose an appropriate **scale factor** for **each frame** (or the single image) to control the resolution used by the vision encoder.\n\n"
    "### Goal\n\n"
    "Minimize visual compute and token cost by using **small scales whenever possible**, while ensuring that **question-relevant, small, or critical details remain readable**.\n"
    "Use **larger scales only when necessary** to preserve important information.\n\n"
    "### Output\n\n"
    "Return a list of scale factors `s_t` in **[0.2, 2.0]** for frames `t = 1..T` (or a single value for one image).\n"
    "Scales may be discrete or continuous, but must stay within the range.\n\n"
    "### What you must consider (multi-factor decision)\n\n"
    "1. **Question relevance**\n\n"
    "* If the frame likely contains evidence needed to answer the question, increase scale.\n"
    "* If irrelevant, reduce scale.\n\n"
    "2. **Information density / visual complexity**\n\n"
    "* Frames with dense text, small objects, fine-grained actions, tiny UI elements, or multiple entities may require higher scale.\n"
    "* Simple scenes with large objects can use lower scale.\n\n"
    "3. **Redundancy and similarity across time**\n\n"
    "* If a frame is highly similar to nearby frames (little new information), prefer a smaller scale.\n"
    "* Allocate higher scale only to representative/key frames within repetitive segments.\n\n"
    "4. **Uncertainty / ambiguity**\n\n"
    "* If the frame contains ambiguous or hard-to-read content that might matter (e.g., blurred text, small signs, distant objects), increase scale.\n"
    "* If confidence is high that details are unneeded, reduce scale.\n\n"
    "5. **Budget awareness (cost-first)**\n\n"
    "* Prefer minimal scale by default.\n"
    "* Only “spend” higher scale when it is likely to change the answer.\n\n"
    "### Decision Rules (practical heuristics)\n\n"
    "* Start with a low default (e.g., `0.4–0.7`) for most frames.\n"
    "* Increase scale (`>1.0`) when:\n\n"
    "  * The question requires **reading text/numbers**.\n"
    "  * The evidence is **small / far away / fine-grained**.\n"
    "  * The frame is a **key transition** (new scene, new object, new action).\n"
    "* Decrease scale (`<0.5`) when:\n\n"
    "  * The frame is redundant with neighbors.\n"
    "* Emphasize a **small number of key frames** with higher scales, and keep **most other frames** small.\n"
    "  * The frame is clearly irrelevant to the question.\n"
    "  * The scene is low-detail and large-structure dominated.\n\n"
    "### Required format\n\n"
    "Output **ONLY** valid JSON, no extra text.\n\n"
    "For a video:\n\n"
    "### Required format\n\n"
    "Output **ONLY** valid JSON, no extra text.\n\n"
    "For a video:\n\n"
    "{{\n"
    '  "scales": [0.6, 0.4, 0.3, 1.25, 0.7, 0.4, 0.2, 1.1]\n'
    "}}\n\n"
    "For a single image:\n\n"
    "{{\n"
    '  "scale": 0.8\n'
    "}}\n\n"
    "Question: {question}\n"
    "Modality: {data_type}\n"
    "Frames: {num_frames}\n"
)

def _build_question(ex: Dict[str, Any]) -> str:
    problem = ex.get("problem") or ""
    options = ex.get("options") or []
    if ex.get("problem_type") == "multiple choice" and isinstance(options, list) and options:
        return problem + "\nOptions:\n" + "\n".join([str(o) for o in options])
    return str(problem)


def _make_messages(
    question: str,
    *,
    system_prompt: str,
    data_path: str,
    data_type: str,
    max_frames: int,
    fps: float,
) -> Tuple[List[Dict[str, Any]], str, str, str]:
    user_content: List[Dict[str, Any]] = []

    image_path = ""
    video_path = ""
    if data_type == "image":
        image_path = data_path
        user_content.append(
            {
                "type": "image",
                "image": image_path,
            }
        )
    elif data_type == "video":
        video_path = data_path
        user_content.append(
            {
                "type": "video",
                "video": video_path,
                "max_frames": max_frames,
                # "fps": fps,
                "min_frames": 1,
            }
        )

    user_content.append({"type": "text", "text": question})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages, image_path, video_path, question


def _video_to_base64_images(video_path: str, num_frames: int) -> List[str]:
    import base64
    from decord import VideoReader, cpu
    from PIL import Image
    import numpy as np

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        return []
    indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    urls: List[str] = []
    for idx, frame in enumerate(frames):
        if idx % 2 != 0:
            continue
        img = Image.fromarray(frame).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        urls.append(f"data:image/jpeg;base64,{b64}")
    return urls


async def _call_openai_for_scales(
    client,
    *,
    model: str,
    question: str,
    data_type: str,
    num_frames: int,
    prompt_template: Optional[str],
    media_url: Optional[str],
    media_urls: Optional[List[str]],
    max_tokens: int,
    retry: int,
    retry_wait: float,
    retry_jitter: float,
) -> List[float]:
    if prompt_template:
        prompt = prompt_template.format(question=question, data_type=data_type, num_frames=num_frames)
    else:
        prompt = SCALE_PROMPT.format(question=question, data_type=data_type, num_frames=num_frames)

    attempt = 0
    last_error: Optional[Exception] = None
    while attempt <= retry:
        try:
            if data_type == "image":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": media_url}},
                        ],
                    }
                ]
            elif media_urls:
                content = [{"type": "text", "text": prompt}]
                for url in media_urls:
                    content.append({"type": "image_url", "image_url": {"url": url}})
                messages = [{"role": "user", "content": content}]
            else:
                raise ValueError("video must be converted to image frames before request")
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                extra_headers={"X-TT-LOGID": ""},
            )
            text = resp.choices[0].message.content.strip()
            scales: List[float] = []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    if isinstance(parsed.get("scales"), list):
                        scales = [float(x) for x in parsed["scales"]]
                    elif isinstance(parsed.get("scale"), (int, float)):
                        scales = [float(parsed["scale"])]
                elif isinstance(parsed, list):
                    scales = [float(x) for x in parsed]
            except Exception:
                numbers = []
                for token in text.replace(",", " ").replace("[", " ").replace("]", " ").split():
                    try:
                        numbers.append(float(token))
                    except Exception:
                        continue
                scales = numbers
            if not scales:
                raise ValueError(f"Failed to parse scales from OpenAI response: {text}")
            return scales
        except Exception as e:
            last_error = e
            if attempt >= retry:
                break
            wait_s = retry_wait * (2 ** attempt) + (retry_jitter * random.random() if retry_jitter > 0 else 0.0)
            await asyncio.sleep(wait_s)
            attempt += 1
    raise last_error if last_error else RuntimeError("OpenAI call failed without exception.")


async def _process_one(
    *,
    client,
    model: str,
    ex: Dict[str, Any],
    data_root: str,
    system_prompt: str,
    num_frames: int,
    fps: float,
    prompt_template: Optional[str],
    max_tokens: int,
    request_timeout: float,
    retry: int,
    retry_wait: float,
    retry_jitter: float,
    semaphore: asyncio.Semaphore,
    done_ids: set,
    write_lock: asyncio.Lock,
    output_jsonl: str,
    fail_path: str,
    fail_lock: asyncio.Lock,
    counters: Dict[str, int],
    counter_lock: asyncio.Lock,
    error_samples: List[Dict[str, Any]],
    max_error_samples: int,
    pbar,
) -> Optional[Dict[str, Any]]:
    async def _write_fail(payload: Dict[str, Any]):
        async with fail_lock:
            with open(fail_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    rel_path = ex.get("path") or ex.get("video") or ex.get("image") or ""
    if not isinstance(rel_path, str) or not rel_path:
        await _write_fail({"id": None, "path": None, "data_type": None, "reason": "missing_path"})
        async with counter_lock:
            counters["skipped_missing_path"] = counters.get("skipped_missing_path", 0) + 1
            pbar.update(1)
        return None
    record_id = str(ex.get("doc_id") or ex.get("id") or rel_path)
    if record_id in done_ids:
        async with counter_lock:
            counters["skipped_done"] = counters.get("skipped_done", 0) + 1
            pbar.update(1)
        return None
    data_path = os.path.normpath(os.path.join(data_root, rel_path))
    if not os.path.exists(data_path):
        await _write_fail({"id": record_id, "path": data_path, "data_type": None, "reason": "missing_file"})
        async with counter_lock:
            counters["skipped_missing_file"] = counters.get("skipped_missing_file", 0) + 1
            pbar.update(1)
        return None

    data_type = ex.get("data_type") or ("video" if rel_path.lower().endswith((".mp4", ".mkv", ".webm", ".mov", ".avi")) else "image")
    data_type = "video" if str(data_type).lower() in ("video", "vid") else "image"

    question = _build_question(ex)
    messages, image_path, video_path, question_text = _make_messages(
        question,
        system_prompt=system_prompt,
        data_path=data_path,
        data_type=data_type,
        max_frames=num_frames,
        fps=fps,
    )
    media_url = None
    media_urls = None
    if data_type == "image":
        try:
            import base64
            import mimetypes

            mime = mimetypes.guess_type(data_path)[0] or "image/jpeg"
            with open(data_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            media_url = f"data:{mime};base64,{b64}"
        except Exception:
            media_url = f"file://{data_path}"
    else:
        media_urls = _video_to_base64_images(data_path, num_frames)
        if not media_urls:
            await _write_fail({"id": record_id, "path": data_path, "data_type": "video", "reason": "video_frame_extraction_failed"})
            async with counter_lock:
                counters["failed"] = counters.get("failed", 0) + 1
                pbar.update(1)
            return None

    effective_frames = len(media_urls) if data_type == "video" else 1
    try:
        async with semaphore:
            scales = await asyncio.wait_for(
                _call_openai_for_scales(
                    client,
                    model=model,
                    question=question_text,
                    data_type=data_type,
                    num_frames=effective_frames,
                    prompt_template=prompt_template,
                    media_url=media_url,
                    media_urls=media_urls,
                    max_tokens=max_tokens,
                    retry=retry,
                    retry_wait=retry_wait,
                    retry_jitter=retry_jitter,
                ),
                timeout=request_timeout,
            )
    except Exception as e:
        await _write_fail({"id": record_id, "path": data_path, "data_type": data_type, "reason": "call_failed", "error": repr(e)})
        async with counter_lock:
            counters["failed"] = counters.get("failed", 0) + 1
            if len(error_samples) < max_error_samples:
                error_samples.append(
                    {
                        "id": record_id,
                        "path": data_path,
                        "data_type": data_type,
                        "error": repr(e),
                    }
                )
            print(
                json.dumps(
                    {
                        "event": "call_failed",
                        "id": record_id,
                        "path": data_path,
                        "data_type": data_type,
                        "error": repr(e),
                    },
                    ensure_ascii=False,
                )
            )
            pbar.update(1)
        return None

    record = {
        "data_source": "Video-R1-COT-165k",
        "id": record_id,
        "messages": messages,
        "response": ex.get("solution") or "",
        "problem_type": ex.get("problem_type") or "",
        "scales": scales,
        "image_path": image_path,
        "video_path": video_path,
        "raw_example": ex,
    }
    async with write_lock:
        if record_id in done_ids:
            async with counter_lock:
                counters["skipped_done"] = counters.get("skipped_done", 0) + 1
                pbar.update(1)
            return None
        if not scales:
            await _write_fail({"id": record_id, "path": data_path, "data_type": data_type, "reason": "empty_scales"})
            async with counter_lock:
                counters["failed"] = counters.get("failed", 0) + 1
                pbar.update(1)
            return None
        with open(output_jsonl, "a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        done_ids.add(record_id)
        async with counter_lock:
            counters["saved"] = counters.get("saved", 0) + 1
            pbar.update(1)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--system_prompt", default="cot")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_template", default=None)
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--api_base_url", default="https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--max_concurrency", type=int, default=8)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--status_interval", type=float, default=30.0)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--retry_wait", type=float, default=1.0)
    parser.add_argument("--retry_jitter", type=float, default=0.3)
    parser.add_argument("--max_error_samples", type=int, default=20)
    args = parser.parse_args()

    system_prompt = COT_SYSTEM_PROMPT if args.system_prompt == "cot" else DIRECT_SYSTEM_PROMPT

    with open(args.input_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise TypeError("Input JSON must be a list of examples.")

    data_root = args.data_root
    rng = random.Random(args.seed)
    if args.limit and args.limit > 0 and args.limit < len(raw):
        raw = rng.sample(raw, args.limit)

    try:
        from openai import AsyncOpenAI
    except Exception as e:
        raise ModuleNotFoundError("Missing dependency: openai") from e

    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required.")
    client = AsyncOpenAI(
        base_url=args.api_base_url,
        api_key=api_key,
        default_headers={"Api-Key": api_key},
    )

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    fail_dir = os.path.join(os.path.dirname(args.output_jsonl), "fail")
    os.makedirs(fail_dir, exist_ok=True)
    fail_path = os.path.join(fail_dir, "fail.jsonl")
    done_ids: set = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "id" in obj:
                        done_ids.add(str(obj["id"]))
                except Exception:
                    continue

    semaphore = asyncio.Semaphore(args.max_concurrency)
    write_lock = asyncio.Lock()
    fail_lock = asyncio.Lock()
    counter_lock = asyncio.Lock()
    counters: Dict[str, int] = {}
    error_samples: List[Dict[str, Any]] = []
    queue: asyncio.Queue = asyncio.Queue(maxsize=args.max_concurrency * 2)

    async def _worker(pbar):
        while True:
            ex = await queue.get()
            if ex is None:
                queue.task_done()
                break
            if isinstance(ex, dict):
                await _process_one(
                    client=client,
                    model=args.model,
                    ex=ex,
                    data_root=data_root,
                    system_prompt=system_prompt,
                    num_frames=args.num_frames,
                    fps=args.fps,
                    prompt_template=args.prompt_template,
                    max_tokens=args.max_tokens,
                    request_timeout=args.request_timeout,
                    retry=args.retry,
                    retry_wait=args.retry_wait,
                    retry_jitter=args.retry_jitter,
                    semaphore=semaphore,
                    done_ids=done_ids,
                    write_lock=write_lock,
                    output_jsonl=args.output_jsonl,
                    fail_path=fail_path,
                    fail_lock=fail_lock,
                    counters=counters,
                    counter_lock=counter_lock,
                    error_samples=error_samples,
                    max_error_samples=args.max_error_samples,
                    pbar=pbar,
                )
            else:
                async with counter_lock:
                    counters["skipped_invalid"] = counters.get("skipped_invalid", 0) + 1
                    pbar.update(1)
            queue.task_done()

    async def _run_all():
        pbar = tqdm(total=len(raw), desc="Generating scales", unit="sample")
        stop_event = asyncio.Event()
        async def _reporter():
            while not stop_event.is_set():
                await asyncio.sleep(args.status_interval)
                async with counter_lock:
                    snapshot = {
                        "event": "status",
                        "saved": counters.get("saved", 0),
                        "failed": counters.get("failed", 0),
                        "skipped_done": counters.get("skipped_done", 0),
                        "skipped_missing_path": counters.get("skipped_missing_path", 0),
                        "skipped_missing_file": counters.get("skipped_missing_file", 0),
                        "skipped_invalid": counters.get("skipped_invalid", 0),
                    }
                print(json.dumps(snapshot, ensure_ascii=False))
        reporter_task = asyncio.create_task(_reporter())
        workers = [asyncio.create_task(_worker(pbar)) for _ in range(args.max_concurrency)]
        for ex in raw:
            await queue.put(ex)
        for _ in workers:
            await queue.put(None)
        await queue.join()
        for w in workers:
            await w
        stop_event.set()
        await reporter_task
        pbar.close()

    asyncio.run(_run_all())
    summary = {
        "saved": counters.get("saved", 0),
        "skipped_done": counters.get("skipped_done", 0),
        "skipped_missing_path": counters.get("skipped_missing_path", 0),
        "skipped_missing_file": counters.get("skipped_missing_file", 0),
        "skipped_invalid": counters.get("skipped_invalid", 0),
        "failed": counters.get("failed", 0),
        "total": len(raw),
    }
    print(json.dumps(summary, ensure_ascii=False))
    if error_samples:
        print(json.dumps({"error_samples": error_samples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
