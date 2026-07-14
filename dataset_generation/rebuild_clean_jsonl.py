#!/usr/bin/env python3
"""
rebuild_clean_jsonl.py
----------------------
Rebuilds a _clean.jsonl from a raw generation output JSONL without calling
any API. Reads every line, keeps only ok=true records, and writes the same
clean schema that write_clean_training_jsonl() produces.

Usage:
    python generation/rebuild_clean_jsonl.py --input <raw.jsonl> [--output <clean.jsonl>]

If --output is omitted, the clean file is written next to the input file
with '_clean' inserted before '.jsonl'.
"""

import argparse
import json
from pathlib import Path

LEARNER_STATE_DEFINITIONS = {
    "Foundational Knowledge Gap": "The learner is missing prerequisite vocabulary, notation, or core concepts required to engage with the current problem or scaffold.",
    "Active Misconception Application": "The learner is actively applying a specific wrong belief or flawed rule to the current problem.",
    "Procedural Execution Bottleneck": "The learner understands the goal but is stuck on a specific mechanical step or calculation.",
    "Task Initiation Hesitation": "The learner is at the start of a problem and lacks a concrete approach, expressing general uncertainty about how to begin.",
    "Spontaneous Conceptual Insight": "The learner has just had a realization or made a connection but has not yet verified or applied it correctly.",
    "Partial Conceptual Grasp": "The learner has grasped part of the concept but is missing a key component needed for full understanding.",
}

FINAL_LEARNER_ROLE_DEFINITIONS = {
    "Broad Help Request": "The learner indicates they cannot proceed or asks for general assistance without identifying a specific sub-task.",
    "Targeted Clarification Request": "The learner specifically asks how to perform a particular operation, substitution, formula use, or intermediate step.",
    "Tentative Answer Proposal": "The learner proposes an answer or next step but signals uncertainty about its correctness.",
    "Independent Reasoning Articulation": "The learner explains their process, summarizes the logic of a step, or verbalizes their current understanding.",
    "Flawed Step Execution": "The learner attempts a specific operation but makes a procedural or conceptual error in executing it.",
}

INTERACTION_PATTERN_DEFINITIONS = {
    "Stepwise Procedural Scaffolding": "The tutor decomposes the problem into smaller sequential execution steps and guides the learner through them.",
    "Reasoning Justification Probe": "The tutor asks the learner to explain why their answer is correct, surfacing hidden gaps or misconceptions.",
    "Hint Clarification Loop": "The tutor gives a targeted hint and then asks the learner to apply it, repeating until the learner demonstrates understanding.",
    "Diagnostic Knowledge Probing": "The tutor asks a series of foundational questions to diagnose where the learner's understanding breaks down.",
    "Direct Error Correction": "The tutor directly identifies and corrects the learner's error, then asks the learner to re-engage with the corrected information.",
    "Premature Answer Jumping": "The tutor attempts to scaffold a substep, but the learner bypasses it by guessing the answer or switching strategies.",
    "Analogical Bridging": "The tutor introduces an analogy or simpler parallel problem to help the learner build intuition before returning to the original task.",
}

TRAJECTORY_MODE_DEFINITIONS = {
    "stays_stuck": "The learner remains confused or unable to proceed despite tutor support.",
    "partial_progress_then_confusion": "The learner makes some progress, then becomes uncertain or confused again.",
    "small_success_no_full_resolution": "The learner achieves a minor success but the core difficulty is not fully resolved.",
    "misunderstanding_repair": "The learner's misconception is identified and partially or fully corrected.",
    "gradual_understanding": "The learner builds understanding step by step, ending with near-complete or complete comprehension.",
}


def rebuild(input_path: str, output_path: str) -> None:
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        return

    if not output_path:
        output_path = str(input_path).replace(".jsonl", "_clean.jsonl")
        if output_path == str(input_path):
            output_path = str(input_path.with_stem(input_path.stem + "_clean"))

    kept = 0
    skipped_failed = 0
    skipped_missing_fields = 0
    skipped_duplicate = 0
    seen_ids = set()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Line {i}: JSON parse error — {e}")
                continue

            if not r.get("ok"):
                skipped_failed += 1
                continue

            if r.get("id") in seen_ids:
                skipped_duplicate += 1
                continue
            seen_ids.add(r.get("id"))

            parsed = r.get("parsed_output")
            spec = r.get("spec")

            if not parsed or not spec:
                skipped_missing_fields += 1
                continue

            try:
                clean_obj = {
                    "id": r["id"],
                    "input_controls": {
                        "subject": spec["subject"],
                        "domain": spec["domain"],
                        "learning_objective": spec["learning_objective"],
                        "target_grade_band": spec["target_grade_band"],
                        "target_stage_in_trajectory": spec["target_stage_in_trajectory"],
                        "target_learner_state": spec["target_learner_state"],
                        "final_learner_role": spec["final_learner_role"],
                        "interaction_pattern": spec["interaction_pattern"],
                        "conversation_setting": spec.get("conversation_setting"),
                        "target_learner_state_definition": LEARNER_STATE_DEFINITIONS.get(spec["target_learner_state"], ""),
                        "final_learner_role_definition": FINAL_LEARNER_ROLE_DEFINITIONS.get(spec["final_learner_role"], ""),
                        "interaction_pattern_definition": INTERACTION_PATTERN_DEFINITIONS.get(spec["interaction_pattern"], ""),
                        "trajectory_mode": spec["trajectory_mode"],
                        "trajectory_mode_definition": TRAJECTORY_MODE_DEFINITIONS.get(spec["trajectory_mode"], ""),
                        "hidden_belief": spec.get("hidden_belief"),
                    },
                    "conversation_setting": parsed["conversation_setting"],
                    "conversation_history": parsed["conversation_history"],
                    "learner_utterance": parsed["learner_utterance"],
                    "latent_state_summary": parsed["latent_state_summary"],
                    "stage_in_trajectory": parsed["stage_in_trajectory"],
                    "learner_state": parsed["learner_state"],
                    "final_learner_role": parsed["final_learner_role"],
                    "interaction_pattern": parsed["interaction_pattern"],
                    "trajectory_mode": parsed["trajectory_mode"],
                }
                fout.write(json.dumps(clean_obj, ensure_ascii=False) + "\n")
                kept += 1
            except KeyError as e:
                print(f"  Line {i}: missing field {e} — skipping")
                skipped_missing_fields += 1

    print(f"Done.")
    print(f"  Kept (ok=true):      {kept}")
    print(f"  Skipped (ok=false):  {skipped_failed}")
    print(f"  Skipped (duplicate): {skipped_duplicate}")
    print(f"  Skipped (bad data):  {skipped_missing_fields}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild _clean.jsonl from raw generation output")
    parser.add_argument("--input",  required=True, help="Path to raw generation JSONL")
    parser.add_argument("--output", default=None,  help="Output path (default: input with _clean suffix)")
    args = parser.parse_args()
    rebuild(args.input, args.output)
