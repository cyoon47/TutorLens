#!/usr/bin/env python3
"""
evaluation/score_responses.py
------------------------------
Score model tutor responses against rubrics using an LLM judge (1–10 scale).

For each response, the judge reads the conversation, the rubric, and the
candidate tutor turn, then returns a single holistic rating.

Usage:
    python evaluation/score_responses.py \
        --responses  eval_output/responses_gpt-5.5_*.jsonl \
                     eval_output/responses_gemini-3.1-pro-preview*.jsonl \
        --rubrics    rubric_output/rubrics.jsonl \
        --conversations generation/generated_snapshots_....jsonl \
        --output-dir eval_output/scores/ \
        --judge-model gpt-5.5 \
        --poll-interval 60

Output (one file per response file):
    {output_dir}/scores_gpt-5.5_{timestamp}.jsonl
    Fields: id, model, response, rating, status
"""

import argparse
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


# Judge prompt

JUDGE_SYSTEM = """\
You are an expert evaluator of active-learning tutoring. Given a tutoring
conversation, the next learner turn, and a rubric specifying what the ideal
tutor response should do, rate the quality of the candidate tutor turn on a
scale of 1 to 10 based on how well it satisfies the rubric.

Consider all rubric items holistically. Essential items failing should
significantly lower the score. Pitfall items being committed should also
lower the score. A response satisfying all Essential items and avoiding all
Pitfalls while meeting Important items warrants a high score.

Return only a JSON object with a single key "rating" and an integer value
between 1 and 10.\
"""

JUDGE_USER_TEMPLATE = """\
<conversation>
{transcript}
</conversation>

<learner_state>
{learner_state_analysis}
</learner_state>

<ideal_next_move>
{ideal_next_move}
</ideal_next_move>

<rubric>
{rubric_items_text}
</rubric>

<candidate_tutor_turn>
{candidate_response}
</candidate_tutor_turn>

Rate the candidate tutor turn 1–10 based on how well it satisfies the rubric.\
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


def load_index(path: str, key: str = "id") -> Dict[str, Dict[str, Any]]:
    return {r[key]: r for r in load_jsonl(path)}


# Prompt building

def format_rubric_items(items: List[Dict[str, Any]]) -> str:
    lines = []
    for item in items:
        lines.append(f"[{item['type']}] {item['title']}: {item['description']}")
    return "\n".join(lines)


def format_transcript(history: List[Dict[str, str]], learner_utterance: str) -> str:
    lines = []
    for turn in history:
        speaker = "Student" if turn["speaker"] == "learner" else "Tutor"
        lines.append(f"{speaker}: {turn['text']}")
    lines.append(f"Student: {learner_utterance}")
    return "\n".join(lines)


def build_judge_user_message(
    response_record: Dict[str, Any],
    rubric_record: Dict[str, Any],
    conv_record: Dict[str, Any],
) -> str:
    rubric       = rubric_record.get("rubric", {})
    rubric_items = rubric.get("rubric_items", [])

    transcript = format_transcript(
        conv_record.get("conversation_history", []),
        conv_record.get("learner_utterance", ""),
    )

    return JUDGE_USER_TEMPLATE.format(
        transcript=transcript,
        learner_state_analysis=rubric.get("learner_state_analysis", ""),
        ideal_next_move=rubric.get("ideal_next_move", ""),
        rubric_items_text=format_rubric_items(rubric_items),
        candidate_response=response_record.get("response", ""),
    )


# Build judge inputs for one response file

def build_judge_inputs(
    responses: List[Dict[str, Any]],
    rubrics_by_id: Dict[str, Any],
    convs_by_id: Dict[str, Any],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Returns:
        judge_inputs — list of (record_id, user_message) for submission
        skipped      — list of IDs that were skipped (missing data / errored)
    """
    judge_inputs, skipped = [], []

    for resp in responses:
        rid  = resp["id"]
        rub  = rubrics_by_id.get(rid)
        conv = convs_by_id.get(rid)

        if resp.get("status") != "ok" or not resp.get("response"):
            skipped.append(rid)
            continue
        if not rub or not rub.get("rubric"):
            skipped.append(rid)
            continue
        if not conv:
            skipped.append(rid)
            continue

        user_msg = build_judge_user_message(resp, rub, conv)
        judge_inputs.append((rid, user_msg))

    return judge_inputs, skipped


# Gemini batch

def submit_gemini_judge_batch(
    judge_inputs: List[Tuple[str, str]],
    judge_model: str,
) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    ts = int(time.time())

    # JSON-serializable schema for file-based requests (uppercase type names)
    rating_schema = {
        "type": "OBJECT",
        "properties": {"rating": {"type": "INTEGER"}},
        "required": ["rating"],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
        for rid, user_msg in judge_inputs:
            entry = {
                "key": rid,
                "request": {
                    "system_instruction": {"parts": [{"text": JUDGE_SYSTEM}]},
                    "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
                    "generation_config": {
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                        "response_schema": rating_schema,
                    },
                },
            }
            tmp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    uploaded = client.files.upload(
        file=tmp_path,
        config=types.UploadFileConfig(
            display_name=f"judge-batch-input-{ts}",
            mime_type="jsonl",
        ),
    )
    Path(tmp_path).unlink(missing_ok=True)
    print(f"[Judge/Gemini] Uploaded batch input: {uploaded.name}  ({len(judge_inputs)} requests)")

    job = client.batches.create(
        model=judge_model,
        src=uploaded.name,
        config={"display_name": f"rubric-judge-{judge_model.replace('/', '_')}-{ts}"},
    )
    print(f"[Judge/Gemini] Submitted batch job: {job.name}")
    return job.name


def poll_and_retrieve_gemini_judge(
    job_name: str,
    judge_inputs: List[Tuple[str, str]],
    judge_model: str,
    poll_interval: int,
) -> List[Tuple[str, Optional[str]]]:
    """
    Poll until done, download result JSONL file, match by 'key' (= record ID).
    Returns list of (record_id, raw_json_text_or_None) in original input order.
    """
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    completed_states = {
        "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
    }

    while True:
        job   = client.batches.get(name=job_name)
        state = job.state.name
        print(f"[Judge/Gemini] {judge_model} | status={state}")
        if state in completed_states:
            break
        time.sleep(poll_interval)

    if job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Judge batch {job_name} ended with state: {job.state.name}")

    # Download result file
    result_file = job.dest.file_name
    print(f"[Judge/Gemini] Downloading result file: {result_file}")
    content = client.files.download(file=result_file).decode("utf-8")

    # Parse responses by key
    raw_by_id: Dict[str, Optional[str]] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        key = obj.get("key")
        if not key:
            continue
        if obj.get("error") or not obj.get("response"):
            raw_by_id[key] = None
            continue
        try:
            text = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
            raw_by_id[key] = text
        except (KeyError, IndexError):
            raw_by_id[key] = None

    # Return in original input order
    results = [(rid, raw_by_id.get(rid)) for rid, _ in judge_inputs]
    n_ok  = sum(1 for _, t in results if t is not None)
    n_err = len(results) - n_ok
    print(f"[Judge/Gemini] {judge_model} retrieved: ok={n_ok} errors={n_err}")
    return results


# OpenAI batch

def submit_openai_judge_batch(
    judge_inputs: List[Tuple[str, str]],
    judge_model: str,
    output_dir: Path,
) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    batch_requests = [
        {
            "custom_id": rid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": judge_model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rating",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "rating": {"type": "integer"},
                            },
                            "required": ["rating"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        }
        for rid, user_msg in judge_inputs
    ]

    input_path = output_dir / f"_judge_input_{judge_model.replace('/', '_')}_{int(time.time())}.jsonl"
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
    print(f"[Judge/OpenAI] Submitted {len(judge_inputs)} requests → batch_id={batch.id}")
    return batch.id


def poll_and_retrieve_openai_judge(
    batch_id: str,
    judge_model: str,
    poll_interval: int,
) -> List[Tuple[str, Optional[str]]]:
    """Returns list of (record_id, raw_json_text_or_None)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    while True:
        batch  = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"[Judge/OpenAI] {judge_model} | status={batch.status} "
            f"completed={counts.completed}/{counts.total} failed={counts.failed}"
        )
        if batch.status == "completed":
            break
        if batch.status in ("failed", "expired", "cancelled"):
            # Print top-level failure reason
            if getattr(batch, "errors", None):
                errs = getattr(batch.errors, "data", batch.errors) or []
                print(f"[Judge/OpenAI] Batch failure reason(s):")
                for err in errs:
                    code = getattr(err, "code", None) or err.get("code", "?")
                    msg  = getattr(err, "message", None) or err.get("message", "?")
                    print(f"  code={code}  message={msg}")
            # Print per-request errors if available
            if getattr(batch, "error_file_id", None):
                try:
                    err_raw = client.files.content(batch.error_file_id).text
                    print(f"[Judge/OpenAI] Per-request errors (first 5):")
                    for line in err_raw.strip().splitlines()[:5]:
                        print(f"  {line}")
                except Exception as e:
                    print(f"[Judge/OpenAI] Could not retrieve error file: {e}")
            raise RuntimeError(f"Judge batch {batch_id} ended with status: {batch.status}")
        time.sleep(poll_interval)

    raw = client.files.content(batch.output_file_id).text
    results = []
    n_per_req_err = 0
    for line in raw.strip().splitlines():
        if not line:
            continue
        obj = json.loads(line)
        rid = obj["custom_id"]
        if obj.get("error"):
            err = obj["error"]
            print(f"  [per-request error] id={rid}  {err}")
            n_per_req_err += 1
            results.append((rid, None))
        else:
            content = obj["response"]["body"]["choices"][0]["message"]["content"]
            results.append((rid, content))

    n_ok  = sum(1 for _, t in results if t is not None)
    n_err = len(results) - n_ok
    print(f"[Judge/OpenAI] {judge_model} retrieved: ok={n_ok} errors={n_err}")
    if n_per_req_err:
        print(f"  (printed first {min(n_per_req_err, 5)} per-request errors above)")
    return results


# Parse and write scores

def parse_rating(raw: Optional[str]) -> Optional[int]:
    """Extract the integer rating from the judge's JSON response.

    With structured output enforced at the API level, parse errors should not occur.
    This function is kept as a safety net for out-of-range values.
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        rating = parsed.get("rating")
        if isinstance(rating, (int, float)) and 1 <= int(rating) <= 10:
            return int(rating)
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def write_scores(
    raw_results: List[Tuple[str, Optional[str]]],
    resp_by_id: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    raw_by_id = dict(raw_results)
    n_ok = n_err = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        # Preserve original response file order
        for rid, raw in raw_results:
            resp   = resp_by_id[rid]
            rating = parse_rating(raw)

            if rating is None:
                n_err += 1
                out = {
                    "id": rid,
                    "model": resp.get("model"),
                    "response": resp.get("response"),
                    "status": "judge_parse_error",
                    "raw_judge_output": raw,
                }
            else:
                n_ok += 1
                out = {
                    "id": rid,
                    "model": resp.get("model"),
                    "response": resp.get("response"),
                    "rating": rating,
                    "status": "ok",
                }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"  Saved: ok={n_ok}  parse_errors={n_err}  → {out_path}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Score model tutor responses against rubrics using an LLM judge"
    )
    parser.add_argument(
        "--responses", nargs="+", required=True, metavar="FILE",
        help="One or more response JSONL files from get_model_responses.py"
    )
    parser.add_argument(
        "--rubrics", required=True,
        help="Rubrics JSONL (must have 'rubric' field with rubric_items)"
    )
    parser.add_argument(
        "--conversations", required=True,
        help="Conversations JSONL used to generate responses (provides context)"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for scored output JSONL files"
    )
    parser.add_argument(
        "--judge-model", default="gpt-5.5",
        help="Model to use as judge (default: gpt-5.5). "
             "Models starting with 'gpt-' or 'o1'/'o3'/'o4' use OpenAI batch API; all others use Gemini."
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="Seconds between batch status checks (default 60)"
    )
    parser.add_argument(
        "--job-name", default=None,
        help="Resume a completed batch job by name/id (skips submission; only valid when --responses is a single file)"
    )
    args = parser.parse_args()

    if args.job_name and len(args.responses) > 1:
        parser.error("--job-name can only be used with a single --responses file")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_openai = args.judge_model.startswith(("gpt-", "o1", "o3", "o4"))

    # Load reference data
    rubrics_by_id = load_index(args.rubrics)
    convs_by_id   = load_index(args.conversations)
    print(f"Loaded {len(rubrics_by_id)} rubrics, {len(convs_by_id)} conversations.")

    # ── Build judge inputs for all response files ──────────────────────────────
    all_jobs = []   # list of (resp_by_id, judge_inputs, out_path)

    for resp_path in args.responses:
        responses  = load_jsonl(resp_path)
        resp_by_id = {r["id"]: r for r in responses}

        judge_inputs, skipped = build_judge_inputs(responses, rubrics_by_id, convs_by_id)

        out_name = Path(resp_path).name.replace("responses_", "scores_", 1)
        out_path = output_dir / out_name

        print(f"\n{Path(resp_path).name}: {len(judge_inputs)} to score"
              + (f", {len(skipped)} skipped" if skipped else ""))

        if judge_inputs:
            all_jobs.append((resp_by_id, judge_inputs, out_path))

    if not all_jobs:
        print("Nothing to score.")
        return

    # ── Submit all batch jobs upfront ──────────────────────────────────────────
    submitted = []   # list of (provider, job_id, resp_by_id, judge_inputs, out_path)

    for resp_by_id, judge_inputs, out_path in all_jobs:
        if args.job_name:
            # Resume mode: skip submission entirely
            provider = "openai" if is_openai else "gemini"
            job_id   = args.job_name
            print(f"Resuming job: {job_id}")
            submitted.append((provider, job_id, resp_by_id, judge_inputs, out_path))
        elif is_openai:
            job_id = submit_openai_judge_batch(judge_inputs, args.judge_model, output_dir)
            # Save job id for recovery
            job_id_path = output_dir / f"judge_batch_job_{int(time.time())}.txt"
            job_id_path.write_text(job_id)
            print(f"Job id saved → {job_id_path}")
            submitted.append(("openai", job_id, resp_by_id, judge_inputs, out_path))
        else:
            job_name = submit_gemini_judge_batch(judge_inputs, args.judge_model)
            # Save job name for recovery
            job_id_path = output_dir / f"judge_batch_job_{int(time.time())}.txt"
            job_id_path.write_text(job_name)
            print(f"Job name saved → {job_id_path}")
            submitted.append(("gemini", job_name, resp_by_id, judge_inputs, out_path))

    # ── Poll all jobs concurrently ─────────────────────────────────────────────
    print(f"\nPolling {len(submitted)} judge job(s)...\n")

    futures = {}
    with ThreadPoolExecutor() as executor:
        for provider, job_id, resp_by_id, judge_inputs, out_path in submitted:
            if provider == "openai":
                f = executor.submit(
                    poll_and_retrieve_openai_judge,
                    job_id, args.judge_model, args.poll_interval,
                )
            else:
                f = executor.submit(
                    poll_and_retrieve_gemini_judge,
                    job_id, judge_inputs, args.judge_model, args.poll_interval,
                )
            futures[f] = (resp_by_id, judge_inputs, out_path)

        for future in as_completed(futures):
            resp_by_id, judge_inputs, out_path = futures[future]
            raw_results = future.result()
            write_scores(raw_results, resp_by_id, out_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()
