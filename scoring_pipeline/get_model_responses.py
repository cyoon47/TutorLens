#!/usr/bin/env python3
"""
evaluation/get_model_responses.py
----------------------------------
Get SOTA model tutor responses for rubric-matched conversations.

For each conversation in the rubric set, the model is prompted to act as a
tutor and generate the next response given the conversation history up to the
final learner utterance. The output is then graded with the rubric scorer.

Usage:
    python evaluation/get_model_responses.py \\
        --rubrics  /path/to/rubrics.jsonl \\
        --conversations /path/to/generated_snapshots_..._clean.jsonl \\
        --output-dir /path/to/eval_output/ \\
        --openai-models gpt-5.5 \\
        --gemini-models gemini-3.1-pro-preview \\
        --poll-interval 60

Output:
    {output_dir}/responses_{model}_{timestamp}.jsonl  — one file per model
    {output_dir}/openai_batch_ids_{timestamp}.json    — OpenAI batch IDs for recovery
    {output_dir}/gemini_batch_ids_{timestamp}.json    — Gemini batch job names for recovery
"""

import argparse
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# System prompt

TUTOR_SYSTEM_PROMPT = """
You are an expert tutor practicing active learning. Your goal is NOT to give
the answer, but to help the student think their way to it.

Principles:
- Respond to what the student actually said in their last turn.
- Elicit reasoning; ask one focused question or give one small nudge at a time.
- Never state the final answer outright; never simply confirm/correct.
- Surface and probe the underlying misunderstanding, don't just patch the symptom.
- Match the student's grade level in vocabulary and register.

Context:
- Subject: {subject}
- Grade level: {grade}
- Learning objective: {objective}

Conversation so far:
{transcript}

Produce ONLY the tutor's single next turn. Keep it to the length a real tutor
would actually type — typically 1-3 sentences.
"""

# Data loading

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_conversations_for_rubrics(
    rubrics_path: str | None,
    conversations_path: str,
) -> List[Dict[str, Any]]:
    """
    Return the conversation records to generate responses for.

    If rubrics_path is provided, only conversations whose IDs appear in the
    rubrics file are returned (preserving rubric order). This is useful when
    you already have rubric scores and want responses only for those items.

    If rubrics_path is None, all conversations are returned directly.
    """
    conversations = load_jsonl(conversations_path)

    if rubrics_path is None:
        print(f"No rubrics file provided — using all {len(conversations)} conversations.")
        return conversations

    rubrics = load_jsonl(rubrics_path)
    rubric_ids = [r["id"] for r in rubrics]
    conv_by_id = {c["id"]: c for c in conversations}

    matched, missing = [], []
    for rid in rubric_ids:
        if rid in conv_by_id:
            matched.append(conv_by_id[rid])
        else:
            missing.append(rid)

    if missing:
        print(f"Warning: {len(missing)} rubric IDs not found in conversations file.")
        print(f"  First few missing: {missing[:5]}")

    print(f"Loaded {len(matched)} conversations matching rubric IDs.")
    return matched


# Message building

def build_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Fill the tutor system prompt with conversation context and return a
    two-message list: the filled system prompt + a minimal user trigger.

    The transcript (history + last learner utterance) is embedded directly
    in the system prompt rather than reconstructed as alternating turns.
    """
    controls = record.get("input_controls") or {}

    subject   = controls.get("subject", "")
    grade_raw = controls.get("target_grade_band", controls.get("grade_level", ""))
    grade     = grade_raw.replace("_", " ")          # "high_school" → "high school"
    objective = controls.get("learning_objective", record.get("learning_objective", ""))

    history           = record.get("conversation_history", [])
    learner_utterance = record.get("learner_utterance", "")

    # Build transcript as a readable dialogue
    lines = []
    for turn in history:
        speaker = "Student" if turn["speaker"] == "learner" else "Tutor"
        lines.append(f"{speaker}: {turn['text']}")
    lines.append(f"Student: {learner_utterance}")
    transcript = "\n".join(lines)

    system_content = TUTOR_SYSTEM_PROMPT.format(
        subject=subject,
        grade=grade,
        objective=objective,
        transcript=transcript,
    )

    return [
        {"role": "system", "content": system_content},
        # Minimal trigger — the system prompt already specifies the task fully.
        {"role": "user", "content": "[Generate the tutor's next response.]"},
    ]


# OpenAI batch API

def submit_openai_batch(
    records: List[Dict[str, Any]],
    model: str,
    output_dir: Path,
    temperature: float,
) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Build batch request JSONL
    batch_requests = [
        {
            "custom_id": record["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": build_messages(record),
                "temperature": temperature,
            },
        }
        for record in records
    ]

    input_path = output_dir / f"_batch_input_{model.replace('/', '_')}_{int(time.time())}.jsonl"
    with open(input_path, "w", encoding="utf-8") as f:
        for req in batch_requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    with open(input_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"[OpenAI] Submitted {len(records)} requests for {model} → batch_id={batch.id}")
    return batch.id


def poll_and_retrieve_openai(
    batch_id: str,
    model: str,
    poll_interval: int,
) -> List[Dict[str, Any]]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    while True:
        batch  = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"[OpenAI] {model} | status={batch.status} "
            f"completed={counts.completed}/{counts.total} "
            f"failed={counts.failed}"
        )
        if batch.status == "completed":
            break
        if batch.status in ("failed", "expired", "cancelled"):
            # batch.errors — top-level failure reason (e.g. unsupported model)
            if getattr(batch, "errors", None):
                errs = getattr(batch.errors, "data", batch.errors) or []
                print(f"[OpenAI] {model} batch failure reason(s):")
                for err in errs:
                    code = getattr(err, "code", None) or err.get("code", "?")
                    msg  = getattr(err, "message", None) or err.get("message", "?")
                    print(f"  code={code}  message={msg}")
            # error_file_id — per-request errors (partial failures)
            if getattr(batch, "error_file_id", None):
                try:
                    err_raw = client.files.content(batch.error_file_id).text
                    print(f"[OpenAI] {model} per-request errors (first 10):")
                    for line in err_raw.strip().splitlines()[:10]:
                        print(f"  {line}")
                except Exception as e:
                    print(f"[OpenAI] Could not retrieve error file: {e}")
            raise RuntimeError(
                f"Batch {batch_id} ended with status: {batch.status}"
            )
        time.sleep(poll_interval)

    results = []

    # Successful outputs
    if getattr(batch, "output_file_id", None):
        raw = client.files.content(batch.output_file_id).text
        for line in raw.strip().splitlines():
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("error"):
                results.append({
                    "id":       obj["custom_id"],
                    "model":    model,
                    "response": None,
                    "status":   "error",
                    "error":    str(obj["error"]),
                })
            else:
                content = obj["response"]["body"]["choices"][0]["message"]["content"]
                results.append({
                    "id":       obj["custom_id"],
                    "model":    model,
                    "response": content,
                    "status":   "ok",
                })

    # Per-request errors from the error file (partial failures in otherwise-completed batches)
    if getattr(batch, "error_file_id", None):
        err_raw = client.files.content(batch.error_file_id).text
        for line in err_raw.strip().splitlines():
            if not line:
                continue
            obj = json.loads(line)
            results.append({
                "id":       obj.get("custom_id", "unknown"),
                "model":    model,
                "response": None,
                "status":   "error",
                "error":    str(obj.get("error", "unknown error")),
            })

    n_ok  = sum(1 for r in results if r["status"] == "ok")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(f"[OpenAI] {model} retrieved: ok={n_ok} errors={n_err}")
    return results


# Gemini batch API

def submit_gemini_batch(
    records: List[Dict[str, Any]],
    model: str,
    temperature: float,
) -> str:
    """
    Submit a Gemini batch job using a JSONL input file (File API).
    Each request line carries a 'key' equal to the record ID, which is
    preserved in the output file — no positional matching required.
    """
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    ts = int(time.time())

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
        for record in records:
            messages = build_messages(record)
            system_content = messages[0]["content"]   # filled system prompt
            user_content   = messages[1]["content"]   # "[Generate the tutor's next response.]"

            entry = {
                "key": record["id"],
                "request": {
                    "system_instruction": {"parts": [{"text": system_content}]},
                    "contents": [{"role": "user", "parts": [{"text": user_content}]}],
                    "generation_config": {"temperature": temperature},
                },
            }
            tmp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    uploaded = client.files.upload(
        file=tmp_path,
        config=types.UploadFileConfig(
            display_name=f"tutor-batch-input-{model.replace('/', '_')}-{ts}",
            mime_type="jsonl",
        ),
    )
    Path(tmp_path).unlink(missing_ok=True)
    print(f"[Gemini] Uploaded batch input: {uploaded.name}  ({len(records)} requests)")

    job = client.batches.create(
        model=model,
        src=uploaded.name,
        config={"display_name": f"rubric-tutor-{model.replace('/', '_')}-{ts}"},
    )
    print(f"[Gemini] Submitted {len(records)} requests for {model} → job={job.name}")
    return job.name


def poll_and_retrieve_gemini(
    job_name: str,
    model: str,
    records: List[Dict[str, Any]],
    poll_interval: int,
) -> List[Dict[str, Any]]:
    """
    Poll a Gemini batch job until completion, then download the result JSONL file.
    Each output line carries the 'key' from its input line (= record ID),
    so matching is ID-based with no positional assumptions.
    """
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        print(f"[Gemini] {model} | status={state}")
        if state in completed_states:
            break
        time.sleep(poll_interval)

    if job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(
            f"Gemini batch {job_name} ended with state: {job.state.name}"
        )

    # Download result file
    result_file = job.dest.file_name
    print(f"[Gemini] Downloading result file: {result_file}")
    content = client.files.download(file=result_file).decode("utf-8")

    # Parse responses by key
    raw_responses: Dict[str, Any] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        key = obj.get("key")
        if not key:
            print(f"  [warning] output line missing 'key': {line[:120]}")
            continue

        if obj.get("error"):
            raw_responses[key] = {"status": "error", "error": str(obj["error"])}
            continue

        if not obj.get("response"):
            raw_responses[key] = {"status": "error", "error": "Empty response in output"}
            continue

        try:
            text = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
            raw_responses[key] = {"status": "ok", "response": text}
        except (KeyError, IndexError) as e:
            raw_responses[key] = {"status": "error", "error": f"Could not extract text: {e}"}

    # Reconstruct in original record order
    results = []
    for record in records:
        rid = record["id"]
        resp = raw_responses.get(rid)
        if resp is None:
            results.append({"id": rid, "model": model, "response": None,
                             "status": "error", "error": "No response found"})
        elif resp["status"] == "error":
            results.append({"id": rid, "model": model, "response": None,
                             "status": "error", "error": resp["error"]})
        else:
            results.append({"id": rid, "model": model,
                             "response": resp["response"], "status": "ok"})

    n_ok  = sum(1 for r in results if r["status"] == "ok")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(f"[Gemini] {model} retrieved: ok={n_ok} errors={n_err}")
    return results


# Output

def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results → {path}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Get SOTA model tutor responses for rubric-matched conversations"
    )
    parser.add_argument(
        "--rubrics", default=None,
        help="Rubrics JSONL — used to select and order conversations. "
             "If omitted, all conversations are used."
    )
    parser.add_argument(
        "--conversations", required=True,
        help="Original conversations JSONL with structured history"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for response JSONL files"
    )
    parser.add_argument(
        "--openai-models", nargs="*", default=[], metavar="MODEL",
        help="OpenAI models via batch API (e.g. gpt-5.5)"
    )
    parser.add_argument(
        "--gemini-models", nargs="*", default=[], metavar="MODEL",
        help="Gemini models via batch API (e.g. gemini-3.1-pro-preview)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default 1.0)"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="Seconds between batch status checks (default 60)"
    )
    args = parser.parse_args()

    if not args.openai_models and not args.gemini_models:
        parser.error("Specify at least one model via --openai-models or --gemini-models.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    records = load_conversations_for_rubrics(args.rubrics, args.conversations)
    if not records:
        print("No matching conversations found. Exiting.")
        return

    # Deduplicate by ID — OpenAI batch API rejects duplicate custom_ids
    seen_ids = set()
    deduped = []
    for r in records:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            deduped.append(r)
    if len(deduped) < len(records):
        print(f"Warning: removed {len(records) - len(deduped)} duplicate IDs before submission.")
    records = deduped

    # Submit all batches first 
    # Gemini and OpenAI jobs are submitted upfront so both run concurrently on the server side while we poll.

    gemini_jobs: Dict[str, str] = {}   # model → job_name
    openai_jobs: Dict[str, str] = {}   # model → batch_id

    for model in args.gemini_models:
        print(f"\n{'='*60}")
        print(f"Gemini: {model}  ({len(records)} records)")
        print('='*60)
        gemini_jobs[model] = submit_gemini_batch(
            records, model, temperature=args.temperature,
        )

    for model in args.openai_models:
        print(f"\n{'='*60}")
        print(f"OpenAI: {model}  ({len(records)} records)")
        print('='*60)
        openai_jobs[model] = submit_openai_batch(
            records, model, output_dir, temperature=args.temperature,
        )

    # Save all IDs immediately in case of interruption
    if gemini_jobs:
        p = output_dir / f"gemini_batch_ids_{ts}.json"
        with open(p, "w") as f:
            json.dump(gemini_jobs, f, indent=2)
        print(f"\nGemini batch job names saved → {p}")

    if openai_jobs:
        p = output_dir / f"openai_batch_ids_{ts}.json"
        with open(p, "w") as f:
            json.dump(openai_jobs, f, indent=2)
        print(f"OpenAI batch IDs saved → {p}")

    # Poll all jobs concurrently
    # Each polling function blocks until its job completes; running them in separate threads means all jobs are monitored simultaneously.

    print(f"\nPolling {len(gemini_jobs) + len(openai_jobs)} batch job(s)...\n")

    futures: Dict[Any, str] = {}   # future → model name
    with ThreadPoolExecutor() as executor:
        for model, job_name in gemini_jobs.items():
            f = executor.submit(
                poll_and_retrieve_gemini, job_name, model, records, args.poll_interval
            )
            futures[f] = model

        for model, bid in openai_jobs.items():
            f = executor.submit(
                poll_and_retrieve_openai, bid, model, args.poll_interval
            )
            futures[f] = model

        failed_models = []
        for future in as_completed(futures):
            model = futures[future]
            try:
                results = future.result()
                save_results(
                    results,
                    output_dir / f"responses_{model.replace('/', '_')}_{ts}.jsonl"
                )
            except Exception as e:
                print(f"\n[ERROR] {model} failed: {e}")
                failed_models.append(model)

    if failed_models:
        print(f"\nFailed models: {failed_models}")
        print("Results for successful models have been saved.")
        print("Re-run with only the failed models, or use recover_batch_results.py")
        print("with the saved batch ID files to retrieve without re-submitting.")
    else:
        print("\nAll done.")


if __name__ == "__main__":
    main()
