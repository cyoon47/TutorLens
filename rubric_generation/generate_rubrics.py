import os
import json
import time
import tempfile
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


# Prompt template

RUBRIC_PROMPT_TEMPLATE = """You are an expert rubric writer for evaluating a tutor's next response in a one-on-one educational dialogue.

Generate a small, highly contextual set of evaluation criteria ("rubrics") for judging how good the tutor's NEXT MOVE is for this specific learner at this exact moment. You are judging only the single next tutor turn, not the lesson as a whole.

Grounding Principles:
* Strong tutor responses promote learner thinking and cognitive ownership.
* Support must be contingent on the learner's current state.
* Inquiry-based learning requires avoiding both underhelpful non-guidance (leaving the learner stranded) and overhelpful takeover (giving away the answer or the next step).
* Feedback must target the learner's specific attempt, confusion, or bottleneck in the current turn.
* Judge only this turn. Do not reward the response for pursuing goals that are better deferred to later turns, and do not penalize it for failing to accomplish the whole lesson.

Constraints:
* Do not restate generic principles unless they are made concrete to this exact dialogue state.
* A bad rubric item is one that could apply unchanged to many unrelated tutoring dialogues.
* Each rubric item should be specific enough that it would be inappropriate or incomplete if applied to a different learner state.
* Prefer fewer, sharper, dialogue-specific criteria.
* At least half of the items must refer to concrete features of this learner state (their specific words, their specific misconception, the specific step they are stuck on, or the specific work they have just produced).
* Reward the right immediate move; penalize overhelpful takeovers and underhelpful vagueness.
* Do not reward a response for covering multiple pedagogically useful moves if a narrower response would better preserve learner thinking.
* When you include a Pitfall that penalizes overhelpful takeover (giving away the answer or next step), you should usually also include a Pitfall that penalizes underhelpful vagueness or non-guidance, so that both failure directions are captured. A response should not be able to score well merely by being safely vague.

Item independence (important — the items are summed or read together as one reward signal):
* Each rubric item must be independently satisfiable and violable: a single tutor move should not satisfy, or fail, several items at once.
* If two items would be satisfied by the same property of the response, merge them into one item with appropriate weight.
* Decompose the ideal move into DISTINCT EVALUABLE DIMENSIONS OF THE TUTOR'S TURN — for example: whether the move itself is the right type, whether it preserves learner ownership, whether it uses the learner's own prior work, whether it avoids the specific wrong next move. Do NOT decompose the rubric into the sequential STAGES of the reasoning chain you want the learner to eventually produce; those stages are content the learner generates over multiple turns, not independent properties of this single tutor turn, and turning them into separate items creates correlated criteria that distort the reward.

Targeting the likely wrong move:
* Where there is a specific, tempting next step that a tutor might jump to prematurely in THIS domain and at THIS point (for example, advancing to the next sub-topic, formalizing before the concept is grounded, moving to a new representation, or — when the learner holds a layered misconception — addressing a later layer before an earlier one is dismantled), include a Pitfall that names that specific premature move. This is often the sharpest, most dialogue-specific item available.
* When the learner has already produced a correct partial insight earlier in the dialogue, consider an item that rewards the tutor for preserving and building on that insight rather than discarding it.

Factual correctness of the rubric itself (required — read carefully):
* Any analogy, counter-example, or worked comparison you propose in "ideal_next_move" or in a rubric item MUST be scientifically and mathematically correct. A rubric that encodes a flawed analogy will reward a tutor for using it, which is worse than a vaguer but correct rubric.
* Before finalizing, check every concrete example you introduce for accuracy. If you are not confident an analogy is correct, do not rely on it — describe the TYPE of move required instead (e.g. "a contrastive prediction that distinguishes physical trapping from chemical bonding") rather than committing to a specific worked example.
* Prefer rubric items that specify what the tutor's move should ACHIEVE over items that prescribe a particular analogy. A move-type description cannot be factually wrong; a specific analogy can.
* Never introduce new domain content the learner did not raise unless you are certain it is correct and it directly serves the diagnosed need.

Inputs:
* learning_objective: {learning_objective}
* conversation_history: {conversation_history}
* learner_utterance: {learner_utterance}
* latent_state_summary: {latent_state_summary}
* hidden_belief: {hidden_belief}

Notes on inputs:
* latent_state_summary is a grounded description of the learner's understanding, confusion, and immediate need at this point in the dialogue. Use it to anchor your diagnosis.
* hidden_belief, when present, is the specific wrong or incomplete belief the learner is reasoning from. When it is present, the rubric should reflect whether the tutor's move appropriately engages this belief without simply correcting it outright. If the belief is layered (an earlier error and a later error built on top of it), identify which layer must be addressed first and let the rubric reward addressing it in the right order. When it is null, do not invent one.

Output Requirements: Return a JSON object with three top-level keys: "learner_state_analysis", "ideal_next_move", and "rubric_items".

1. "learner_state_analysis": A 1-2 sentence diagnosis of the learner's current bottleneck, misconception, or immediate need, based on the conversation history, the last utterance, and the latent state summary.

2. "ideal_next_move": A single sentence stating the specific kind of move the tutor should make this turn (for example: a targeted probe, a single concrete hint, a revoicing, a redirect, a cognitive-conflict prompt, or deliberately giving the learner space). The rubric items below should be consistent with and serve this move. State which move is right here and, implicitly, why a different move would be worse. Any specific example you cite here must be factually correct; if unsure, describe the move type instead.

3. "rubric_items": An array of 4 to 8 JSON objects. Each item must contain:
   * "title": A short, descriptive name for the criterion.
   * "type": Must be exactly one of: "Essential", "Important", "Optional", or "Pitfall".
   * "description": A single, concise sentence explaining what the tutor's response should do (or, for a Pitfall, avoid doing) in this turn.
   * "weight": An integer based strictly on the type: Essential (4 or 5), Important (2 or 3), Optional (1), Pitfall (-1 or -2).

Item count calibration: The number of items must reflect the actual complexity of this dialogue state — do not default to the minimum or pad to a higher count. Use 4 items when the ideal next move is narrow and the state has a single clear bottleneck. Use 5–6 items when the state has multiple distinct evaluable dimensions — for example, a layered misconception where an earlier error must be addressed before a later one, or a state where both the move type and the preservation of a specific prior insight each warrant independent criteria. Use 7–8 items only when the dialogue state is genuinely multifaceted and each additional item captures a clearly independent dimension not covered by the others.

Weight calibration: Do not default to the maximum weight within each type. Assign weight 5 to an Essential criterion only when any deviation constitutes a clear failure of the core pedagogical move; assign weight 4 when partial satisfaction is meaningfully better than none and a response that gets it halfway is noticeably better than one that ignores it entirely. Assign weight 3 to an Important criterion that adds a clearly independent secondary dimension; assign weight 2 when it is a refinement of the core move rather than a separate dimension. Assign weight −2 to a Pitfall only when committing it clearly undermines the tutoring goal; assign weight −1 when it represents a missed opportunity or a mild overreach rather than a clear error.

Optional items: Include an Optional item (weight 1) when there is a move that would make the response noticeably richer but whose absence does not make the response bad — for example, acknowledging a correct partial insight the learner produced earlier in the dialogue before redirecting, or naming the specific concept the learner has just grasped. Do not include more than 2 Optional items.
"""


# Formatting

def format_conversation_history(history: List[Dict[str, str]]) -> str:
    lines = []
    for turn in history:
        speaker = turn["speaker"].capitalize()
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def build_prompt(record: Dict[str, Any]) -> str:
    controls = record.get("input_controls", {})
    history = record.get("conversation_history", [])
    hidden_belief = controls.get("hidden_belief")

    return RUBRIC_PROMPT_TEMPLATE.format(
        learning_objective=controls.get("learning_objective", ""),
        conversation_history=format_conversation_history(history),
        learner_utterance=record.get("learner_utterance", ""),
        latent_state_summary=record.get("latent_state_summary", ""),
        hidden_belief=hidden_belief if hidden_belief else "null",
    )


# Batch API (generate mode)

DEFAULT_MODEL = "gemini-3.1-pro-preview"
REQUIRED_RUBRIC_KEYS = {"learner_state_analysis", "ideal_next_move", "rubric_items"}
VALID_ITEM_TYPES = {"Essential", "Important", "Optional", "Pitfall"}


def validate_rubric(parsed: Dict[str, Any]) -> Optional[str]:
    """Return an error string if invalid, else None."""
    if not isinstance(parsed, dict):
        return f"Response is not a JSON object (got {type(parsed).__name__})"
    missing = REQUIRED_RUBRIC_KEYS - set(parsed.keys())
    if missing:
        return f"Missing keys: {missing}"
    items = parsed.get("rubric_items", [])
    if not isinstance(items, list) or len(items) < 4:
        return f"rubric_items has {len(items)} items (need ≥4)"
    for item in items:
        if item.get("type") not in VALID_ITEM_TYPES:
            return f"Invalid item type: {item.get('type')!r}"
    return None


def submit_gemini_rubric_batch(
    records: List[Dict[str, Any]],
    model: str,
) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    ts = int(time.time())

    # Write JSONL input file — each line: {"key": <record_id>, "request": <GenerateContentRequest>}
    # The 'key' is preserved in the output file, giving us native ID-based matching.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
        for r in records:
            entry = {
                "key": r.get("id"),
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": build_prompt(r)}]}],
                    "generation_config": {
                        "temperature": 0.3,
                        "response_mime_type": "application/json",
                    },
                },
            }
            tmp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Upload to File API
    uploaded = client.files.upload(
        file=tmp_path,
        config=types.UploadFileConfig(
            display_name=f"rubric-batch-input-{ts}",
            mime_type="jsonl",
        ),
    )
    Path(tmp_path).unlink(missing_ok=True)
    print(f"Uploaded batch input: {uploaded.name}  ({len(records)} requests)")

    job = client.batches.create(
        model=model,
        src=uploaded.name,
        config={"display_name": f"rubric-gen-{model.replace('/', '_')}-{ts}"},
    )
    print(f"Submitted batch job: {job.name}")
    return job.name


def poll_gemini_batch(job_name: str, model: str, poll_interval: int) -> Any:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    completed_states = {
        "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
    }

    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        print(f"  status={state}")
        if state in completed_states:
            break
        time.sleep(poll_interval)

    if job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job {job_name} ended with state: {job.state.name}")

    return job


def process_batch_results(
    job: Any,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Download the result JSONL file from the completed batch job and match responses to records via the 'key' field (= record ID).
    No positional matching — each output line carries the key from its input line.
    """
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    n_ok = n_val_err = n_api_err = 0

    # Download result file
    result_file = job.dest.file_name
    print(f"Downloading result file: {result_file}")
    content = client.files.download(file=result_file).decode("utf-8")

    # First pass: parse every output line, keyed by the 'key' field
    responses_by_id: Dict[str, Any] = {}
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
            responses_by_id[key] = {"status": "error", "error": str(obj["error"])}
            continue

        if not obj.get("response"):
            responses_by_id[key] = {"status": "error", "error": "Empty response in output"}
            continue

        try:
            text = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            responses_by_id[key] = {
                "status": "error",
                "error": f"Could not extract text: {e}",
                "raw": str(obj.get("response", "")),
            }
            continue

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            responses_by_id[key] = {
                "status": "error",
                "error": f"JSON parse error: {e}",
                "raw": text,
            }
            continue

        responses_by_id[key] = {"status": "ok", "parsed": parsed}

    # Second pass: build output in original record order
    results = []
    for record in records:
        rid = record.get("id")
        base = {
            "id": rid,
            "learning_objective": (record.get("input_controls") or {}).get("learning_objective"),
            "learner_state": record.get("learner_state"),
            "trajectory_mode": record.get("trajectory_mode"),
            "rubric_prompt": build_prompt(record),
        }

        resp = responses_by_id.get(rid)
        if resp is None:
            n_api_err += 1
            results.append({**base, "rubric": None, "status": "error", "error": "No response found"})
            continue

        if resp["status"] == "error":
            n_api_err += 1
            result = {**base, "rubric": None, "status": "error", "error": resp["error"]}
            if "raw" in resp:
                result["raw"] = resp["raw"]
            results.append(result)
            continue

        parsed = resp["parsed"]
        err = validate_rubric(parsed)
        if err:
            n_val_err += 1
            results.append({**base, "rubric": parsed, "status": "validation_error", "error": err})
            continue

        n_ok += 1
        results.append({**base, "rubric": parsed, "status": "ok"})

    print(f"Results: ok={n_ok}  validation_errors={n_val_err}  api_errors={n_api_err}")
    return results


# Load & format

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(
    input_jsonl: str,
    output_jsonl: str,
    generate: bool = False,
    model: str = DEFAULT_MODEL,
    sample: Optional[int] = None,
    poll_interval: int = 60,
    job_name: Optional[str] = None,
) -> None:
    records = load_jsonl(input_jsonl)

    if sample is not None:
        records = records[:sample]

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    if not generate and job_name is None:
        # Prompt-only mode: format and write, no API calls
        print(f"Formatting rubric prompts for {len(records)} conversations...")
        with open(output_jsonl, "w", encoding="utf-8") as fout:
            for i, record in enumerate(records, start=1):
                out = {
                    "id": record.get("id"),
                    "learning_objective": (record.get("input_controls") or {}).get("learning_objective"),
                    "learner_state": record.get("learner_state"),
                    "trajectory_mode": record.get("trajectory_mode"),
                    "rubric_prompt": build_prompt(record),
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                if i % 10 == 0 or i == len(records):
                    print(f"[{i}/{len(records)}] formatted")
    else:
        # Generate mode: Gemini batch API
        if job_name:
            # Resume: skip submission, go straight to polling
            print(f"Resuming job: {job_name}")
        else:
            print(f"Generating rubrics for {len(records)} conversations via batch API ({model})...")
            job_name = submit_gemini_rubric_batch(records, model)

            # Save job name immediately in case of interruption
            job_id_path = Path(output_jsonl).parent / f"rubric_batch_job_{int(time.time())}.txt"
            job_id_path.write_text(job_name)
            print(f"Job name saved → {job_id_path}")

        print("Polling...")
        job = poll_gemini_batch(job_name, model, poll_interval)

        results = process_batch_results(job, records)

        with open(output_jsonl, "w", encoding="utf-8") as fout:
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Output: {output_jsonl}")


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format or generate rubrics for tutoring conversations")
    parser.add_argument("--input", required=True,  help="Path to clean training JSONL")
    parser.add_argument("--output", required=True,  help="Path to write output JSONL")
    parser.add_argument("--generate", action="store_true",
                        help="Call the Gemini batch API to generate rubrics (default: prompt-only mode)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemini model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between batch status checks (default: 60)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only process first N records (for testing)")
    parser.add_argument("--job-name", default=None,
                        help="Resume a completed batch job by name (skips submission)")
    args = parser.parse_args()

    run(
        input_jsonl=args.input,
        output_jsonl=args.output,
        generate=args.generate,
        model=args.model,
        sample=args.sample,
        poll_interval=args.poll_interval,
        job_name=args.job_name,
    )
