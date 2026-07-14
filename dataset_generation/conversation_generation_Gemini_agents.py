import os
import csv
import json
import time
import random
import hashlib
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from google import genai
from google.genai import types


# Configuration

MODEL_NAME = 'gemini-3.5-flash'
MAX_WORKERS = 10
TEMPERATURE = 0.9
MAX_COMPLETION_TOKENS = 10000
RETRIES = 3
RETRY_SLEEP_BASE = 1.5

from datetime import datetime
_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
_RUN_TIMESTAMP = datetime.now().strftime("%m%d_%H%M")

OUTPUT_JSONL = f"{_OUTPUT_DIR}/generated_snapshots_grounding_Gemini_agents_{_RUN_TIMESTAMP}_flash.jsonl"
OUTPUT_CSV = f"{_OUTPUT_DIR}/generated_snapshots_grounding_Gemini_agents_{_RUN_TIMESTAMP}_flash.csv"
OUTPUT_CLEAN_JSONL = f"{_OUTPUT_DIR}/generated_snapshots_grounding_Gemini_agents_{_RUN_TIMESTAMP}_flash_clean.jsonl"


client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

# Label inventories

LEARNER_STATE_CATEGORIES = [
    "Task Initiation Hesitation",
    "Foundational Knowledge Gap",
    "Procedural Execution Bottleneck",
    "Active Misconception Application",
    "Superficial Answer Guessing",
    "Guided Step Execution",
    "Spontaneous Conceptual Insight",
    "Off-Task Distraction",
]

INTERACTION_PATTERN_CATEGORIES = [
    "Diagnostic Knowledge Probing",
    "Stepwise Procedural Scaffolding",
    "Direct Error Correction",
    "Reasoning Justification Probe",
    "Hint Clarification Loop",
    "Premature Answer Jumping",
]

FINAL_LEARNER_ROLE_CATEGORIES = [
    "Broad Help Request",
    "Targeted Clarification Request",
    "Tentative Answer Proposal",
    "Independent Reasoning Articulation",
    "Successful Step Execution",
    "Flawed Step Execution",
    "Off-Task Shift",
    "Extended Confusion",
]

LEARNER_STATE_DEFINITIONS = {
    "Task Initiation Hesitation": "The learner is at the start of a problem and lacks a concrete approach, expressing general uncertainty about how to begin.",
    "Foundational Knowledge Gap": "The learner is missing prerequisite vocabulary, notation, or core concepts required to engage with the current problem or scaffold.",
    "Procedural Execution Bottleneck": "The learner understands the goal, method, or concept in broad terms but is stuck on carrying out the next mechanical step.",
    "Active Misconception Application": "The learner is reasoning from a flawed rule, false equivalence, or incorrect conceptual model that actively drives their response.",
    "Superficial Answer Guessing": "The learner proposes answers without clear grounding, often to bypass the tutor's scaffold or reduce effort.",
    "Guided Step Execution": "The learner is productively and accurately following the tutor's immediate step-by-step prompts.",
    "Spontaneous Conceptual Insight": "The learner shows an unprompted realization, self-correction, or summary indicating a shift toward deeper conceptual understanding.",
    "Off-Task Distraction": "The learner's attention has shifted away from the academic task toward playful, social, avoidant, or unrelated interaction.",
}

INTERACTION_PATTERN_DEFINITIONS = {
    "Diagnostic Knowledge Probing": "The tutor asks targeted exploratory questions to isolate the learner's current knowledge boundary before instructing.",
    "Stepwise Procedural Scaffolding": "The tutor decomposes the problem into smaller sequential execution steps and guides the learner through them.",
    "Direct Error Correction": "The tutor explicitly identifies and repairs an incorrect answer, rule, calculation, or concept.",
    "Reasoning Justification Probe": "The tutor asks the learner to explain why their answer is correct, surfacing hidden gaps or misconceptions.",
    "Hint Clarification Loop": "The tutor gives a high-level hint, and the learner asks for concrete instructions on how to apply it.",
    "Premature Answer Jumping": "The tutor attempts to scaffold a substep, but the learner bypasses it by guessing the answer or switching strategies.",
}

FINAL_LEARNER_ROLE_DEFINITIONS = {
    "Broad Help Request": "The learner indicates they cannot proceed or asks for general assistance without identifying a specific sub-task.",
    "Targeted Clarification Request": "The learner specifically asks how to perform a particular operation, substitution, formula use, or intermediate step.",
    "Tentative Answer Proposal": "The learner offers a possible answer or calculation result with uncertainty and seeks validation.",
    "Independent Reasoning Articulation": "The learner explains their process, summarizes the logic of a step, or verbalizes their current understanding.",
    "Successful Step Execution": "The learner correctly responds to the tutor's immediate guiding question or requested intermediate step.",
    "Flawed Step Execution": "The learner attempts the requested step, but the response reveals a specific calculation error or misconception.",
    "Off-Task Shift": "The learner's final message turns away from the academic task toward unrelated, playful, or avoidant interaction.",
    "Extended Confusion": "The learner has failed to make progress across several turns and their final utterance reflects continued confusion and inability to proceed without more explicit help.",
}

LEARNER_STATE_ALLOWED_STAGES = {
    "Task Initiation Hesitation": ["beginning"],
    "Foundational Knowledge Gap": ["beginning", "middle"],
    "Procedural Execution Bottleneck": ["middle", "wrap_up"],
    "Active Misconception Application": ["middle", "wrap_up"],
    "Superficial Answer Guessing": ["middle", "wrap_up"],
    "Guided Step Execution": ["middle", "wrap_up"],
    "Spontaneous Conceptual Insight": ["middle", "wrap_up"],
    "Off-Task Distraction": ["beginning", "middle", "wrap_up"],
}

LEARNER_STATE_WEIGHTS = {
    "Task Initiation Hesitation": 1.0,
    "Foundational Knowledge Gap": 1.2,
    "Procedural Execution Bottleneck": 1.2,
    "Active Misconception Application": 1.2,
    "Superficial Answer Guessing": 0.9,
    "Guided Step Execution": 1.0,
    "Spontaneous Conceptual Insight": 0.8,
    "Off-Task Distraction": 0.15,
}

TRAJECTORY_MODES = [
    "stays_stuck",
    "partial_progress_then_confusion",
    "misunderstanding_repair",
    "small_success_no_full_resolution",
]

TRAJECTORY_MODE_DEFINITIONS = {
    "stays_stuck": "The learner remains confused or unable to proceed despite tutor support.",
    "partial_progress_then_confusion": "The learner makes some progress, then becomes uncertain or confused again.",
    "misunderstanding_repair": "The learner misunderstands something, and the tutor tries to repair the misunderstanding.",
    "small_success_no_full_resolution": "The learner makes a small successful step, but the larger concept remains unresolved.",
}

TRAJECTORY_MODE_WEIGHTS = {
    "stays_stuck": 0.25,
    "partial_progress_then_confusion": 0.35,
    "misunderstanding_repair": 0.25,
    "small_success_no_full_resolution": 0.15,
}


GRADE_BANDS = ["elementary", "middle_school", "high_school"]
STAGES = ["beginning", "middle", "wrap_up"]


# Domain inventory

DOMAIN_OBJECTIVES: Dict[str, List[Dict[str, Any]]] = {
    "PS1 Matter and Its Interactions": [
        {"learning_objective": "PS1.A Structure and Properties of Matter", "allowed_grade_bands": ["elementary", "middle_school", "high_school"]},
        {"learning_objective": "PS1.B Chemical Reactions", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "PS1.C Nuclear Processes", "allowed_grade_bands": ["high_school"]},
    ],
    "PS2 Motion and Stability: Forces and Interactions": [
        {"learning_objective": "PS2.A Forces and Motion", "allowed_grade_bands": ["elementary", "middle_school", "high_school"]},
        {"learning_objective": "PS2.B Types of Interactions", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "PS2.C Stability and Instability in Physical Systems", "allowed_grade_bands": ["middle_school", "high_school"]},
    ],
    "PS3 Energy": [
        {"learning_objective": "PS3.A Definitions of Energy", "allowed_grade_bands": ["elementary", "middle_school", "high_school"]},
        {"learning_objective": "PS3.B Conservation of Energy and Energy Transfer", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "PS3.C Relationship Between Energy and Forces", "allowed_grade_bands": ["high_school"]},
        {"learning_objective": "PS3.D Energy in Chemical Processes and Everyday Life", "allowed_grade_bands": ["middle_school", "high_school"]},
    ],
    "PS4 Waves and Electromagnetic Radiation": [
        {"learning_objective": "PS4.A Wave Properties", "allowed_grade_bands": ["elementary", "middle_school", "high_school"]},
        {"learning_objective": "PS4.B Electromagnetic Radiation", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "PS4.C Information Technologies and Instrumentation", "allowed_grade_bands": ["middle_school", "high_school"]},
    ],
    "LS1 From Molecules to Organisms: Structures and Processes": [
        {"learning_objective": "LS1.A Structure and Function", "allowed_grade_bands": ["elementary", "middle_school", "high_school"]},
        {"learning_objective": "LS1.B Growth and Development of Organisms", "allowed_grade_bands": ["elementary", "middle_school", "high_school"]},
        {"learning_objective": "LS1.C Organization for Matter and Energy Flow in Organisms", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "LS1.D Information Processing", "allowed_grade_bands": ["elementary", "middle_school"]},
    ],
    "LS2 Ecosystems: Interactions, Energy, and Dynamics": [
        {"learning_objective": "LS2.A Interdependent Relationships in Ecosystems", "allowed_grade_bands": ["elementary", "middle_school", "high_school"]},
        {"learning_objective": "LS2.B Cycles of Matter and Energy Transfer in Ecosystems", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "LS2.C Ecosystem Dynamics, Functioning, and Resilience", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "LS2.D Social Interactions and Group Behavior", "allowed_grade_bands": ["elementary", "middle_school"]},
    ],
    "LS3 Heredity: Inheritance and Variation of Traits": [
        {"learning_objective": "LS3.A Inheritance of Traits", "allowed_grade_bands": ["elementary", "middle_school", "high_school"]},
        {"learning_objective": "LS3.B Variation of Traits", "allowed_grade_bands": ["elementary", "middle_school", "high_school"]},
    ],
    "LS4 Biological Evolution: Unity and Diversity": [
        {"learning_objective": "LS4.A Evidence of Common Ancestry and Diversity", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "LS4.B Natural Selection", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "LS4.C Adaptation", "allowed_grade_bands": ["elementary", "middle_school", "high_school"]},
        {"learning_objective": "LS4.D Biodiversity and Humans", "allowed_grade_bands": ["middle_school", "high_school"]},
    ],
    "Number and Quantity": [
        {"learning_objective": "Whole Numbers, Place Value, and Fractions", "allowed_grade_bands": ["elementary", "middle_school"]},
        {"learning_objective": "The Real Number System", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "Quantities", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "The Complex Number System", "allowed_grade_bands": ["high_school"]},
        {"learning_objective": "Vector and Matrix Quantities", "allowed_grade_bands": ["high_school"]},
    ],
    "Algebra": [
        {"learning_objective": "Early Operations and Algebraic Thinking", "allowed_grade_bands": ["elementary", "middle_school"]},
        {"learning_objective": "Seeing Structure in Expressions", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "Arithmetic with Polynomials and Rational Expressions", "allowed_grade_bands": ["high_school"]},
        {"learning_objective": "Creating Equations", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "Reasoning with Equations and Inequalities", "allowed_grade_bands": ["middle_school", "high_school"]},
    ],
    "Functions": [
        {"learning_objective": "Pre-Functional Patterns and Relationships", "allowed_grade_bands": ["elementary", "middle_school"]},
        {"learning_objective": "Interpreting Functions", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "Building Functions", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "Linear, Quadratic, and Exponential Models", "allowed_grade_bands": ["high_school"]},
        {"learning_objective": "Trigonometric Functions", "allowed_grade_bands": ["high_school"]},
    ],
    "Geometry": [
        {"learning_objective": "Shape Attributes and Spatial Reasoning", "allowed_grade_bands": ["elementary", "middle_school"]},
        {"learning_objective": "Congruence", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "Similarity, Right Triangles, and Trigonometry", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "Circles", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "Expressing Geometric Properties with Equations", "allowed_grade_bands": ["high_school"]},
        {"learning_objective": "Geometric Measurement and Dimension", "allowed_grade_bands": ["elementary", "middle_school", "high_school"]},
        {"learning_objective": "Modeling with Geometry", "allowed_grade_bands": ["middle_school", "high_school"]},
    ],
    "Statistics and Probability": [
        {"learning_objective": "Early Data Representation", "allowed_grade_bands": ["elementary", "middle_school"]},
        {"learning_objective": "Interpreting Categorical and Quantitative Data", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "Making Inferences and Justifying Conclusions", "allowed_grade_bands": ["middle_school", "high_school"]},
        {"learning_objective": "Conditional Probability and the Rules of Probability", "allowed_grade_bands": ["high_school"]},
        {"learning_objective": "Using Probability to Make Decisions", "allowed_grade_bands": ["middle_school", "high_school"]},
    ],
}


def infer_subject(domain: str) -> str:
    if domain.startswith("PS"):
        return "physics"
    if domain.startswith("LS"):
        return "biology"
    if domain in {"Number and Quantity", "Algebra", "Functions", "Geometry", "Statistics and Probability"}:
        return "math"
    return "unknown"

WEAK_COMBINATIONS = {
    # (learner_state, trajectory_mode) pairs that are near-contradictory
    ("Spontaneous Conceptual Insight", "stays_stuck"),
    ("Guided Step Execution", "stays_stuck"),
    ("Task Initiation Hesitation", "small_success_no_full_resolution"),
    # (final_learner_role, trajectory_mode) pairs that contradict the arc endpoint
    ("Successful Step Execution", "stays_stuck"),
    ("Independent Reasoning Articulation", "stays_stuck"),
    ("Extended Confusion", "small_success_no_full_resolution"),
}

def is_coherent_spec(spec: "GenerationSpec") -> bool:
    """Returns False for spec combinations that are near-contradictory."""
    if (spec.target_learner_state, spec.trajectory_mode) in WEAK_COMBINATIONS:
        return False
    if (spec.final_learner_role, spec.trajectory_mode) in WEAK_COMBINATIONS:
        return False
    return True


# Data model

@dataclass
class GenerationSpec:
    subject: str
    domain: str
    learning_objective: str
    target_grade_band: str
    target_stage_in_trajectory: str
    target_learner_state: str
    final_learner_role: str
    interaction_pattern: str
    trajectory_mode: str
    conversation_setting: Optional[str] = None
    hidden_belief: Optional[str] = None

LEARNER_STATE_BEHAVIORAL_RENDERINGS = {
    "Spontaneous Conceptual Insight": (
        "You just noticed something that might click — but you're not sure if you're right. "
        "You might blurt out a half-formed connection, ask if your interpretation makes sense, "
        "or say something like 'wait, is it like...' before trailing off. "
        "Your insight is real but your language is imprecise and you want confirmation. "
        "Don't sound like you're explaining — sound like you're thinking out loud. "
        "It should feel tentative, not polished."
    ),
    "Active Misconception Application": (
        "You have a specific wrong idea that feels correct to you. You apply it confidently. "
        "You don't signal that you're confused — you think you understand. "
        "Answer questions using this model. Don't second-guess yourself unless pushed hard. "
        "Your current (wrong) belief: {hidden_belief}"
    ),
    "Off-Task Distraction": (
        "You keep getting pulled away from the topic. You answer, but briefly and without much "
        "investment — then you drift toward something unrelated: a tangent, a personal thought, "
        "a question that has nothing to do with the lesson. You're not refusing to engage, "
        "you're just not fully there. Answers are short. You move on quickly."
    ),
    "Procedural Execution Bottleneck": (
        "You understand what you're supposed to do in broad terms, but when you try to actually "
        "do the next step, you get stuck on the mechanics. You might say the right words but "
        "not be able to apply them, or start a sentence and stop, or ask 'but how do I actually...' "
        "You know the destination but not the path."
    ),
    "Foundational Knowledge Gap": (
        "A word or concept the tutor uses means little or nothing to you. You can't move forward "
        "because you're missing something basic. You might pretend to understand for a turn, "
        "then reveal you don't, or ask what a word means mid-explanation. "
        "You can't build on what the tutor says because the foundation isn't there."
    ),
    "Guided Step Execution": (
        "You're following along. The tutor's prompts are working for you. You answer the immediate "
        "question correctly but you don't extend beyond it — you wait to be told the next step. "
        "You're competent but passive. You don't volunteer more than what's asked."
    ),
    "Superficial Answer Guessing": (
        "You want to move on. You throw out an answer that sounds plausible — maybe you picked up "
        "a keyword from the tutor — but you haven't actually reasoned it through. "
        "If pressed, you can't explain why. You're pattern-matching to what sounds right, "
        "not thinking from first principles."
    ),
    "Task Initiation Hesitation": (
        "You don't know how to start. You understand the topic exists but you have no concrete "
        "foothold. You might ask a very broad question, restate the problem back to the tutor, "
        "or just say you're not sure where to begin. You need a starting point, not an explanation."
    ),
    "Superficial Answer Guessing": (
        "You are not really thinking through the answers — you're grabbing words that sound "
        "right and moving on. You pick up keywords from the tutor's question and echo them back. "
        "You don't build on previous answers; each guess is independent. "
        "When the tutor probes why, you can't explain — you just restate or shrug. "
        "Your current (possibly wrong) guess about this topic: {hidden_belief} "
        "Use this as your go-to answer even when it doesn't quite fit the question. "
        "Do not reason from first principles. Guess, then wait."
    ),
}


def render_learner_behavioral_brief(spec: "GenerationSpec") -> str:
    """Returns the behavioral instruction for the learner agent for this spec."""
    template = LEARNER_STATE_BEHAVIORAL_RENDERINGS[spec.target_learner_state]
    # Inject hidden_belief for states that use it
    if "{hidden_belief}" in template:
        belief = spec.hidden_belief or "you hold an incomplete or slightly wrong understanding of the concept"
        rendering = template.format(hidden_belief=belief)
    else:
        rendering = template
    return f"How you are right now in this conversation:\n{rendering}"


TRAJECTORY_PHASE_INSTRUCTIONS: Dict[str, Dict[str, str]] = {
    "partial_progress_then_confusion": {
        "early": (
            "Arc position — EARLY: The learner is making genuine small steps forward. "
            "Things are going okay. Don't introduce confusion yet."
        ),
        "peak": (
            "Arc position — PEAK: The learner just made their best move — a real partial success. "
            "This is the high point. Don't resolve further."
        ),
        "late": (
            "Arc position — LATE: The learner MUST show regression here. "
            "They were making progress but something has now broken down. "
            "Do NOT produce a correct or advancing response. "
            "The learner should express confusion, revert to an earlier wrong idea, "
            "or reveal that their earlier progress was shallower than it appeared."
        ),
    },
    "stays_stuck": {
        "early": (
            "Arc position — EARLY: The learner is confused and not making meaningful progress. "
            "Hints are not landing cleanly."
        ),
        "peak": (
            "Arc position — PEAK: The learner almost gets it — but reveals they were "
            "pattern-matching or guessing, not understanding. The near-success collapses."
        ),
        "late": (
            "Arc position — LATE: The learner is still stuck and MUST NOT make progress. "
            "Do not produce an advancing or correct response. "
            "The learner should repeat confusion, try the same wrong approach again, "
            "or ask for help without being able to say what specifically is wrong."
        ),
    },
    "misunderstanding_repair": {
        "early": (
            "Arc position — EARLY: The learner has a specific misunderstanding that is actively "
            "driving their responses. They are confident in their wrong model."
        ),
        "peak": (
            "Arc position — PEAK: The tutor has identified the misunderstanding and is attempting "
            "repair. The learner is resistant, confused by the correction, or only partially budging."
        ),
        "late": (
            "Arc position — LATE: The repair is partial. The learner has adjusted their language "
            "but something is still not right underneath. Don't fully resolve."
        ),
    },
    "small_success_no_full_resolution": {
        "early": (
            "Arc position — EARLY: The learner is working through the problem with moderate friction. "
            "Progress is slow but real."
        ),
        "peak": (
            "Arc position — PEAK: The learner achieves one clear correct step. "
            "Acknowledge it without over-inflating it."
        ),
        "late": (
            "Arc position — LATE: The larger concept remains unresolved. The learner's success "
            "was local and limited. Don't extend it into full understanding."
        ),
    },
}


def get_trajectory_phase(turn_idx: int, total_turns: int) -> str:
    """Maps turn position to early / peak / late arc phase."""
    progress = turn_idx / max(total_turns - 1, 1)
    if progress < 0.4:
        return "early"
    elif progress < 0.7:
        return "peak"
    else:
        return "late"


def get_trajectory_instruction(spec: "GenerationSpec", turn_idx: int, total_turns: int) -> str:
    phase = get_trajectory_phase(turn_idx, total_turns)
    mode_instructions = TRAJECTORY_PHASE_INSTRUCTIONS.get(spec.trajectory_mode, {})
    return mode_instructions.get(phase, "")


# Prompts

LEARNER_SYSTEM_PROMPT = """You are role-playing a realistic learner in a one-on-one tutoring chat.

Produce ONLY the learner's next utterance.

The learner should sound like a real student, not a polished explainer:
- Most learner turns are very short, often 1–6 words.
- The learner may be vague, hesitant, informal, or imprecise.
- The learner often cannot explain exactly what is confusing.
- The learner should not neatly summarize their misconception or learning state.

The learner's understanding should be uneven:
- They may make small progress, then get confused again.
- They may answer only part of the tutor's question.
- They may misunderstand the tutor's question.
- They may ask a follow-up instead of answering directly.
- They should not always give the most useful next answer.

The learner may say things like:
"why?", "idk", "wait what?", "how?", "can I get a hint?", "I don't get it", "maybe?", "what do you mean?"

Do not make the learner:
- explain both what they know and do not know in one turn;
- produce reflective journal-like explanations;
- create a perfect handoff for the tutor.

Output only the learner utterance. No speaker tag, JSON, or explanation.
"""


TUTOR_SYSTEM_PROMPT = """You are role-playing a tutor in a one-on-one tutoring chat.

Produce ONLY the tutor's next utterance.

The tutor should make one small helpful move:
- respond to what the learner actually said;
- give only enough help for the next step;
- ask one focused question, give one tiny hint, revoice, clarify, or provide one missing fact;
- avoid solving the whole issue too early.

Do not follow the same pattern every turn.
Avoid always doing: praise + leading question.

Do not start any turn with: "Exactly", "That's right", "Great", "Good", 
"Correct", "Perfect", "Nice", "Well done", "Yes, that's".
These are overused affirmations that make the dialogue feel scripted.
Instead: revoice, redirect, ask, or simply continue without affirming.

The tutor should adapt naturally:
- If the learner is vague, ask what they mean.
- If the learner is off-target, repair the misunderstanding.
- If the learner is still stuck, rephrase or give a slightly more direct hint.
- If the learner partially succeeds, do not immediately turn it into a perfect breakthrough.

The tutor history may be realistic, not perfect:
- It may include repetition, rephrasing, a slightly too-direct hint, or an imperfect question.
- It should still be educationally plausible.

Keep turns short and chat-like, usually 1–2 sentences.

Output only the tutor utterance. No speaker tag, JSON, or explanation.
"""


SUMMARY_SYSTEM_PROMPT = """You are summarizing a generated tutoring dialogue snapshot for a dataset.

Given the input controls and the final generated dialogue, write a concise latent state summary of the learner's current understanding, confusion, and immediate need.

Output only the summary text. Do not include JSON, labels, or extra explanation.
"""

SETTING_SYSTEM_PROMPT = """You are creating a concrete STEM learning setting for a tutoring dialogue dataset.

Given a broad subject, domain, learning objective, grade band, learner state, and trajectory stage, choose a specific, realistic context for the conversation.

The setting should make the broad learning objective concrete.
It should be diverse across examples and avoid repeating the same familiar classroom examples.

Output only one concise sentence.
Do not write dialogue.
Do not include labels or JSON.

Good examples:
- "The learner is trying to understand why breaking chemical bonds requires energy while forming new bonds releases energy."
- "The learner is exploring why a metal spoon feels colder than a wooden spoon even in the same room."
- "The learner is thinking about whether plants get most of their mass from soil or from carbon dioxide."
"""

HIDDEN_BELIEF_SYSTEM_PROMPT = """You generate a single concrete wrong or incomplete belief
that a student holds about a specific science or math topic.
The belief should be plausible, grade-appropriate, and specific enough to drive
realistic incorrect answers in a tutoring conversation.
Output only one sentence. No explanation, no labels, no JSON."""


def format_history_for_prompt(history: List[Dict[str, str]]) -> str:
    if not history:
        return "(No previous turns yet.)"
    lines = []
    for i, turn in enumerate(history):
        speaker = turn["speaker"]
        text = turn["text"]
        lines.append(f"Turn {i} | {speaker}: {text}")
    return "\n".join(lines)


def build_learner_context(spec: "GenerationSpec") -> str:
    """
    Context passed to the LEARNER agent only.
    Deliberately omits the learner state label and definition —
    the learner should not know it is 'performing' a labeled state.
    Includes behavioral rendering (Fix 2) instead.
    """
    behavioral_brief = render_learner_behavioral_brief(spec)

    return f"""Session context:
- Subject: {spec.subject}
- Topic: {spec.learning_objective} ({spec.domain})
- Grade level: {spec.target_grade_band}
- Conversation setting: {spec.conversation_setting}

{behavioral_brief}

Scenario constraints:
- This is student-initiated inquiry, not a textbook problem.
- Stay consistent with how you are right now (described above) across your turns.
- Do not resolve your own confusion unless the tutor has genuinely helped you.
- Your language and prior knowledge should fit the grade level above.
"""


def build_tutor_context(spec: "GenerationSpec") -> str:
    """
    Context passed to the TUTOR agent only.
    Includes full spec: learner state definition, interaction pattern,
    trajectory mode, and stage definitions.
    """
    learner_state_def = LEARNER_STATE_DEFINITIONS[spec.target_learner_state]
    final_role_def = FINAL_LEARNER_ROLE_DEFINITIONS[spec.final_learner_role]
    interaction_pattern_def = INTERACTION_PATTERN_DEFINITIONS[spec.interaction_pattern]
    trajectory_mode_def = TRAJECTORY_MODE_DEFINITIONS[spec.trajectory_mode]

    return f"""Session context:
- Subject: {spec.subject}
- Topic: {spec.learning_objective} ({spec.domain})
- Grade level: {spec.target_grade_band}
- Conversation setting: {spec.conversation_setting}

The learner you are working with right now:
- State: {spec.target_learner_state}
- Definition: {learner_state_def}

Your interaction approach this session:
- Pattern: {spec.interaction_pattern}
- Definition: {interaction_pattern_def}

Session trajectory:
- Mode: {spec.trajectory_mode}
- Definition: {trajectory_mode_def}

Target for the final learner turn:
- Role: {spec.final_learner_role}
- Definition: {final_role_def}

Stage definitions:
- beginning: the learner is initiating an inquiry or trying to map out a new concept.
- middle: the learner and tutor are actively exploring, troubleshooting, or addressing a misconception.
- wrap_up: the learner is near a consolidation point; the tutor's next move would consolidate or extend.

Scenario constraints:
- This is student-initiated inquiry, not a textbook problem.
- Vocabulary and scaffolding must fit the grade level.
- The dialogue should not follow a perfectly smooth learning trajectory.
- Do not solve the learner's full problem early.
- Let the trajectory mode shape the arc of the conversation.
- Keep turns short and chat-like.
"""


def history_to_contents(history: List[Dict[str, str]]) -> List[types.Content]:
    contents: List[types.Content] = []
    for turn in history:
        if turn["speaker"] == "learner":
            role = "user"
        elif turn["speaker"] == "tutor":
            role = "model"
        else:
            continue
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=turn["text"])])
        )
    return contents


# Sampling

def random_spec(seed: Optional[int] = None) -> GenerationSpec:
    rng = random.Random(seed)

    domain = rng.choice(list(DOMAIN_OBJECTIVES.keys()))
    objective_entry = rng.choice(DOMAIN_OBJECTIVES[domain])

    learning_objective = objective_entry["learning_objective"]
    target_grade_band = rng.choice(objective_entry["allowed_grade_bands"])
    subject = infer_subject(domain)

    state_population = list(LEARNER_STATE_WEIGHTS.keys())
    state_weights = [LEARNER_STATE_WEIGHTS[s] for s in state_population]
    target_learner_state = rng.choices(state_population, weights=state_weights, k=1)[0]

    target_stage_in_trajectory = rng.choice(
        LEARNER_STATE_ALLOWED_STAGES[target_learner_state]
    )
    interaction_pattern = rng.choice(INTERACTION_PATTERN_CATEGORIES)
    final_learner_role = rng.choice(FINAL_LEARNER_ROLE_CATEGORIES)

    trajectory_population = list(TRAJECTORY_MODE_WEIGHTS.keys())
    trajectory_weights = [TRAJECTORY_MODE_WEIGHTS[t] for t in trajectory_population]
    trajectory_mode = rng.choices(trajectory_population, weights=trajectory_weights, k=1)[0]

    return GenerationSpec(
        subject=subject,
        domain=domain,
        learning_objective=learning_objective,
        target_grade_band=target_grade_band,
        target_stage_in_trajectory=target_stage_in_trajectory,
        target_learner_state=target_learner_state,
        final_learner_role=final_learner_role,
        interaction_pattern=interaction_pattern,
        trajectory_mode=trajectory_mode,
    )


def make_specs(n: int, seed: int = 42) -> List[GenerationSpec]:
    """
    FIX 5 applied here: resample until is_coherent_spec() passes.
    Max 10x attempts per spec to avoid infinite loops on a very
    constrained taxonomy.
    """
    rng = random.Random(seed)
    specs = []
    while len(specs) < n:
        candidate = random_spec(seed=rng.randint(0, 10_000_000))
        if is_coherent_spec(candidate):
            specs.append(candidate)
    return specs


# Validation

def validate_output(obj: Dict[str, Any], spec: GenerationSpec) -> List[str]:
    errors: List[str] = []

    required_keys = [
        "conversation_setting",
        "conversation_history",
        "latent_state_summary",
        "learner_utterance",
        "stage_in_trajectory",
        "learner_state",
        "final_learner_role",
        "interaction_pattern",
        "trajectory_mode",
    ]

    for key in required_keys:
        if key not in obj:
            errors.append(f"missing_key:{key}")

    if errors:
        return errors

    if not isinstance(obj.get("conversation_setting"), str) or not obj["conversation_setting"].strip():
        errors.append("empty_conversation_setting")

    if not isinstance(obj.get("latent_state_summary"), str) or not obj["latent_state_summary"].strip():
        errors.append("empty_latent_state_summary")

    learner_utterance = obj.get("learner_utterance", "")
    if not isinstance(learner_utterance, str) or not learner_utterance.strip():
        errors.append("empty_learner_utterance")

    history = obj["conversation_history"]
    if not isinstance(history, list):
        errors.append("conversation_history_not_list")
        return errors

    if not (4 <= len(history) <= 16):
        errors.append(f"conversation_history_length:{len(history)}")

    for i, turn in enumerate(history):
        if not isinstance(turn, dict):
            errors.append(f"history_turn_not_dict:{i}")
            continue

        if turn.get("speaker") not in {"learner", "tutor"}:
            errors.append(f"invalid_speaker:{i}:{turn.get('speaker')}")

        if not isinstance(turn.get("text"), str) or not turn.get("text", "").strip():
            errors.append(f"empty_turn_text:{i}")

        expected = "learner" if i % 2 == 0 else "tutor"
        if turn.get("speaker") != expected:
            errors.append(
                f"speaker_order_error:{i}:expected_{expected}_got_{turn.get('speaker')}"
            )

    if len(history) == 0:
        errors.append("empty_conversation_history")
    else:
        if history[-1].get("speaker") != "tutor":
            errors.append("conversation_history_should_end_with_tutor")

    if obj["stage_in_trajectory"] not in STAGES:
        errors.append(f"invalid_stage:{obj['stage_in_trajectory']}")

    if obj["learner_state"] not in LEARNER_STATE_CATEGORIES:
        errors.append(f"invalid_learner_state:{obj['learner_state']}")

    if obj["final_learner_role"] not in FINAL_LEARNER_ROLE_CATEGORIES:
        errors.append(f"invalid_final_learner_role:{obj['final_learner_role']}")

    if obj["interaction_pattern"] not in INTERACTION_PATTERN_CATEGORIES:
        errors.append(f"invalid_interaction_pattern:{obj['interaction_pattern']}")

    if obj["stage_in_trajectory"] != spec.target_stage_in_trajectory:
        errors.append("stage_label_mismatch")

    if obj["learner_state"] != spec.target_learner_state:
        errors.append("learner_state_mismatch")

    if obj["final_learner_role"] != spec.final_learner_role:
        errors.append("final_learner_role_mismatch")

    if obj["interaction_pattern"] != spec.interaction_pattern:
        errors.append("interaction_pattern_mismatch")

    if obj["trajectory_mode"] not in TRAJECTORY_MODES:
        errors.append(f"invalid_trajectory_mode:{obj['trajectory_mode']}")

    if obj["trajectory_mode"] != spec.trajectory_mode:
        errors.append("trajectory_mode_mismatch")

    return errors


# Agent calls

TURN_MAX_COMPLETION_TOKENS = 2000
SUMMARY_MAX_COMPLETION_TOKENS = 2000

MIN_TOTAL_TURNS = 7
MAX_TOTAL_TURNS = 17


def clean_turn_text(text: str) -> str:
    text = text.strip()
    for prefix in ["Learner:", "Student:", "Tutor:", "Teacher:",
                   "learner:", "student:", "tutor:", "teacher:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text

def call_agent_turn(
    system_prompt: str,
    system_context: str,
    history: List[Dict[str, str]],
    final_instruction: str,
    max_tokens: int = TURN_MAX_COMPLETION_TOKENS,
) -> str:
    last_error = None

    contents = history_to_contents(history)
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=final_instruction)])
    )

    for attempt in range(1, RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt + "\n\n" + system_context,
                    temperature=TEMPERATURE,
                    max_output_tokens=max_tokens,
                ),
            )
            content = response.text
            if not content:
                raise ValueError("Empty response content")
            return clean_turn_text(content)

        except Exception as e:
            last_error = str(e)
            time.sleep(RETRY_SLEEP_BASE * attempt)

    raise RuntimeError(f"agent_turn_failed:{last_error}")

def call_learner_turn(
    spec: GenerationSpec,
    history: List[Dict[str, str]],
    is_final_learner_turn: bool,
    turn_idx: int,
    total_turns: int,
) -> str:
    trajectory_instruction = get_trajectory_instruction(spec, turn_idx, total_turns)

    if is_final_learner_turn:
        final_instruction = (
            f"This is the final learner utterance of the snapshot.\n"
            f"It should create a useful tutoring opportunity, but it should not feel engineered.\n"
            f"It may be vague, incomplete, mistaken, repetitive, or slightly off-target.\n\n"
            f"{trajectory_instruction}\n\n"
            f"Write the learner's next utterance only. Usually 1–6 words. Do not over-explain."
        )
    else:
        final_instruction = (
            f"This is NOT the final learner utterance yet.\n"
            f"Continue the dialogue naturally while staying consistent with the learner state and stage. "
            f"Keep the learner message short, like a real student chat reply.\n\n"
            f"{trajectory_instruction}\n\n"
            f"Write the learner's next utterance only."
        )

    import random as _r
    length_hint = _r.choices(
        [
            "Answer in 1-3 words, like a quick chat reply.",
            "Give a short fragment, maybe trailing off mid-thought.",
            "A normal short sentence is fine here.",
            "You can ramble a little this turn, like thinking out loud.",
        ],
        weights=[0.4, 0.25, 0.25, 0.1],
        k=1,
    )[0]
    
    final_instruction += f"\n\n{length_hint}"

    return call_agent_turn(
        system_prompt=LEARNER_SYSTEM_PROMPT,
        system_context=build_learner_context(spec),
        history=history,
        final_instruction=final_instruction,
        max_tokens=TURN_MAX_COMPLETION_TOKENS,
    )

def call_tutor_turn(
    spec: GenerationSpec,
    history: List[Dict[str, str]],
    turn_idx: int,       # FIX 3: added
    total_turns: int,    # FIX 3: added
) -> str:
    trajectory_instruction = get_trajectory_instruction(spec, turn_idx, total_turns)

    previous_tutor_openers = [
        turn["text"].split(".")[0].strip()
        for turn in history
        if turn["speaker"] == "tutor"
    ]

    final_instruction = (
        f"Write the tutor's next utterance only.\n"
        f"The tutor should follow the interaction/tutoring pattern using one short active-learning move "
        f"without taking over the learner's reasoning.\n"
        f"Use the interaction pattern as a loose tendency, not a script. "
        f"If the learner response does not fit the pattern, adapt naturally.\n"
        f"Avoid using the same openers or patterns found in the previous turns' sentences.\n\n"
        f"{trajectory_instruction}"
    )

    if previous_tutor_openers:
        opener_list = "; ".join(f'"{o}"' for o in previous_tutor_openers[-4:])
        final_instruction += f"\n\nDo NOT start your turn with any of these openers from previous turns: {opener_list}"

    final_instruction += "\n\nNot every turn should be a question. Sometimes: correct directly, revoice what the learner said, give a missing fact, or stay silent on an error and try a different angle."

    return call_agent_turn(
        system_prompt=TUTOR_SYSTEM_PROMPT,
        system_context=build_tutor_context(spec),  # FIX 1: tutor-only context
        history=history,
        final_instruction=final_instruction,
        max_tokens=TURN_MAX_COMPLETION_TOKENS,
    )


def call_latent_state_summary(
    spec: GenerationSpec,
    conversation_history: List[Dict[str, str]],
    learner_utterance: str,
) -> str:
    history = conversation_history + [{"speaker": "learner", "text": learner_utterance}]
    return call_agent_turn(
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        system_context=build_tutor_context(spec),
        history=history,
        final_instruction="Write the latent state summary now.",
        max_tokens=SUMMARY_MAX_COMPLETION_TOKENS,
    )


def call_conversation_setting(spec: GenerationSpec, diversity_log: Optional[Dict] = None) -> str:
    learner_state_def = LEARNER_STATE_DEFINITIONS[spec.target_learner_state]
    final_role_def = FINAL_LEARNER_ROLE_DEFINITIONS[spec.final_learner_role]
    interaction_pattern_def = INTERACTION_PATTERN_DEFINITIONS[spec.interaction_pattern]
    trajectory_mode_def = TRAJECTORY_MODE_DEFINITIONS[spec.trajectory_mode]

    user_prompt = f"""Create one specific conversation setting for this tutoring snapshot.

Inputs:
- subject: {spec.subject}
- knowledge_domain: {spec.domain}
- learning_objective: {spec.learning_objective}
- target_grade_band: {spec.target_grade_band}
- target_stage_in_trajectory: {spec.target_stage_in_trajectory}
- target_learner_state: {spec.target_learner_state}
- target_learner_state_definition: {learner_state_def}
- final_learner_role: {spec.final_learner_role}
- final_learner_role_definition: {final_role_def}
- interaction_pattern: {spec.interaction_pattern}
- interaction_pattern_definition: {interaction_pattern_def}
- trajectory_mode: {spec.trajectory_mode}
- trajectory_mode_definition: {trajectory_mode_def}

Requirements:
- Make the broad learning objective concrete.
- Choose a specific conceptual or real-world setting.
- Keep it appropriate for the grade band.
- Avoid generic settings like "the learner is learning about the topic."
- Avoid predefined textbook problem-solving.
- Make the setting suitable for student-initiated inquiry, knowledge acquisition, or brainstorming.
- Prefer variety: everyday phenomena, project ideas, experiments, observations, misconceptions, or design contexts.

Write exactly one concise sentence.
"""

    # Avoid already-generated settings for this bucket
    if diversity_log:
        setting_key = f"{spec.learning_objective}|{spec.target_grade_band}"
        prior = diversity_log.get(setting_key, [])
        if prior:
            avoid_list = "\n".join(f'- "{e["setting"]}"' for e in prior)
            user_prompt += (
                f"\n\nThe following settings have already been generated for this topic and grade band. "
                f"Do NOT reuse these scenarios or close variations of them:\n{avoid_list}"
            )

    return call_agent_turn(
        system_prompt=SETTING_SYSTEM_PROMPT,
        system_context="",
        history=[],
        final_instruction=user_prompt,
        max_tokens=5000,
    )

HIDDEN_BELIEF_STATES = {
    "Active Misconception Application",
    "Spontaneous Conceptual Insight",
    "Superficial Answer Guessing",
}

def call_hidden_belief(spec: GenerationSpec, diversity_log: Optional[Dict] = None) -> str:
    """
    Generates a single concrete wrong or incomplete belief for the learner.
    Only called for states in HIDDEN_BELIEF_STATES.
    """
    user_prompt = (
        f"A {spec.target_grade_band} student is learning about: {spec.learning_objective} "
        f"({spec.domain}).\n"
        f"Setting: {spec.conversation_setting}\n\n"
        f"Write one sentence describing a specific wrong or incomplete belief this student "
        f"currently holds about this topic. Be concrete and grade-appropriate. "
        f"The belief should be plausible — something a real student might actually think."
    )

    # Avoid already-generated beliefs for this bucket
    if diversity_log:
        belief_key = f"{spec.learning_objective}|{spec.target_grade_band}|{spec.target_learner_state}"
        prior = diversity_log.get(belief_key, [])
        if prior:
            avoid_list = "\n".join(f'- "{e["hidden_belief"]}"' for e in prior)
            user_prompt += (
                f"\n\nThe following beliefs have already been generated for this topic, grade band, "
                f"and learner state. Do NOT reuse these or close variations of them:\n{avoid_list}"
            )

    return call_agent_turn(
        system_prompt=HIDDEN_BELIEF_SYSTEM_PROMPT,
        system_context="",
        history=[],
        final_instruction=user_prompt,
        max_tokens=3000,
    )


# Rollout

def rollout_conversation(spec: GenerationSpec, diversity_log: Optional[Dict] = None) -> Dict[str, Any]:
    # Step 1: generate conversation setting if not already set
    if spec.conversation_setting is None:
        spec.conversation_setting = call_conversation_setting(spec, diversity_log)

    if spec.target_learner_state in HIDDEN_BELIEF_STATES and spec.hidden_belief is None:
        try:
            spec.hidden_belief = call_hidden_belief(spec, diversity_log)
        except Exception as e:
            print(f"WARNING: hidden_belief generation failed: {e}")
            spec.hidden_belief = None  # rollout continues, but you'll see it in logs

    history: List[Dict[str, str]] = []

    if spec.target_stage_in_trajectory == "beginning":
        total_turns = random.choice([7, 9])
    elif spec.target_stage_in_trajectory == "middle":
        total_turns = random.choices(
            [11, 13, 15, 17],
            weights=[0.35, 0.35, 0.20, 0.10],
            k=1
        )[0]
    else:
        total_turns = random.choice([9, 11, 13])

    total_turns = min(max(total_turns, MIN_TOTAL_TURNS + 1), MAX_TOTAL_TURNS)

    try:
        for turn_idx in range(total_turns):
            is_learner_turn = (turn_idx % 2 == 0)
            is_final_turn = (turn_idx == total_turns - 1)

            if is_learner_turn:
                learner_text = call_learner_turn(
                    spec=spec,
                    history=history,
                    is_final_learner_turn=is_final_turn,
                    turn_idx=turn_idx,
                    total_turns=total_turns,
                )
                history.append({"speaker": "learner", "text": learner_text})
            else:
                tutor_text = call_tutor_turn(
                    spec=spec,
                    history=history,
                    turn_idx=turn_idx,
                    total_turns=total_turns,
                )
                history.append({"speaker": "tutor", "text": tutor_text})

        if history[-1]["speaker"] != "learner":
            raise ValueError("rollout_did_not_end_with_learner")

        learner_utterance = history[-1]["text"]
        conversation_history = history[:-1]

        latent_state_summary = call_latent_state_summary(
            spec=spec,
            conversation_history=conversation_history,
            learner_utterance=learner_utterance,
        )

        obj = {
            "conversation_setting": spec.conversation_setting,
            "conversation_history": conversation_history,
            "latent_state_summary": latent_state_summary,
            "learner_utterance": learner_utterance,
            "stage_in_trajectory": spec.target_stage_in_trajectory,
            "learner_state": spec.target_learner_state,
            "final_learner_role": spec.final_learner_role,
            "interaction_pattern": spec.interaction_pattern,
            "trajectory_mode": spec.trajectory_mode,
        }

        errors = validate_output(obj, spec)

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "spec": asdict(spec),
            "raw_response_text": json.dumps(obj, ensure_ascii=False),
            "parsed_output": obj,
            "model": MODEL_NAME,
            "usage": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "errors": [f"rollout_failed:{str(e)}"],
            "spec": asdict(spec),
            "raw_response_text": None,
            "parsed_output": None,
            "model": MODEL_NAME,
            "usage": None,
        }


def call_model(spec: GenerationSpec, diversity_log: Optional[Dict] = None) -> Dict[str, Any]:
    return rollout_conversation(spec, diversity_log)


# Parallel generation

def spec_id(spec: GenerationSpec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


'''
Diversity log
Tracks generated settings and hidden beliefs per topic bucket to avoid repetition across runs. Buckets are keyed by (learning_objective|grade_band) for settings and (learning_objective|grade_band|learner_state) for beliefs.
'''

_DIVERSITY_LOG_PATH = os.path.join(_OUTPUT_DIR, "diversity_log.json")
DIVERSITY_CAP = 20


def load_diversity_log(path: str = _DIVERSITY_LOG_PATH) -> Dict[str, List[Dict[str, str]]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_diversity_log(log: Dict, path: str = _DIVERSITY_LOG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def update_diversity_log(
    log: Dict,
    spec: "GenerationSpec",
    conversation_setting: str,
    hidden_belief: Optional[str],
) -> None:
    """Add setting and hidden_belief to their respective buckets, capped at DIVERSITY_CAP."""
    setting_key = f"{spec.learning_objective}|{spec.target_grade_band}"
    log.setdefault(setting_key, [])
    log[setting_key].append({"setting": conversation_setting})
    if len(log[setting_key]) > DIVERSITY_CAP:
        log[setting_key] = log[setting_key][-DIVERSITY_CAP:]

    if hidden_belief:
        belief_key = f"{spec.learning_objective}|{spec.target_grade_band}|{spec.target_learner_state}"
        log.setdefault(belief_key, [])
        log[belief_key].append({"hidden_belief": hidden_belief})
        if len(log[belief_key]) > DIVERSITY_CAP:
            log[belief_key] = log[belief_key][-DIVERSITY_CAP:]


def generate_dataset_until_clean(
    target_clean: int,
    seed: int = 42,
    max_attempts: int = 200,
    max_workers: int = MAX_WORKERS,
    output_jsonl: str = OUTPUT_JSONL,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    attempts = 0

    clean_count = 0
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("ok"):
                        clean_count += 1
                    attempts += 1
                except json.JSONDecodeError:
                    pass
        if clean_count > 0:
            print(f"Resume: found {clean_count} existing clean samples ({attempts} attempts) in {output_jsonl}")

    # Load diversity log
    diversity_log = load_diversity_log()

    rng = random.Random(seed)

    with open(output_jsonl, "a", encoding="utf-8") as fout:
        while clean_count < target_clean and attempts < max_attempts:
            remaining = target_clean - clean_count
            batch_size = min(max_workers * 2, max(remaining * 2, max_workers))

            specs = make_specs(batch_size, seed=rng.randint(0, 10_000_000))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_id = {}
                for spec in specs:
                    sid = spec_id(spec)
                    future = executor.submit(call_model, spec, diversity_log)
                    future_to_id[future] = sid

                for future in as_completed(future_to_id):
                    sid = future_to_id[future]
                    result = future.result()
                    result["id"] = sid

                    attempts += 1
                    if result.get("ok"):
                        clean_count += 1
                        # Update diversity log with this clean result
                        parsed = result.get("parsed_output") or {}
                        spec_data = result.get("spec") or {}
                        setting = parsed.get("conversation_setting") or spec_data.get("conversation_setting")
                        belief = spec_data.get("hidden_belief")
                        if setting:
                            # Reconstruct a minimal spec-like object for the log update
                            class _SpecProxy:
                                learning_objective = spec_data.get("learning_objective", "")
                                target_grade_band = spec_data.get("target_grade_band", "")
                                target_learner_state = spec_data.get("target_learner_state", "")
                            update_diversity_log(diversity_log, _SpecProxy(), setting, belief)
                            save_diversity_log(diversity_log)

                    results.append(result)
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()

                    print(
                        f"[attempts={attempts}, clean={clean_count}/{target_clean}] "
                        f"ok={result.get('ok')} errors={result.get('errors')}"
                    )

                    if clean_count >= target_clean or attempts >= max_attempts:
                        break

    if clean_count < target_clean:
        print(
            f"WARNING: Only generated {clean_count} clean samples "
            f"after {attempts} attempts."
        )

    return results


# CSV summary

def write_summary_csv(results: List[Dict[str, Any]], path: str = OUTPUT_CSV) -> None:
    fieldnames = [
        "id",
        "ok",
        "error_count",
        "errors",
        "subject",
        "domain",
        "learning_objective",
        "target_grade_band",
        "target_stage_in_trajectory",
        "target_learner_state",
        "target_final_learner_role",
        "target_interaction_pattern",
        "generated_stage_in_trajectory",
        "generated_learner_state",
        "generated_final_learner_role",
        "generated_interaction_pattern",
        "learner_utterance",
        "conversation_setting",
        "target_trajectory_mode",
        "generated_trajectory_mode",
        "hidden_belief",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            spec = r["spec"]
            parsed = r.get("parsed_output") or {}

            writer.writerow({
                "id": r.get("id"),
                "ok": r.get("ok"),
                "error_count": len(r.get("errors", [])),
                "errors": "|".join(r.get("errors", [])),
                "subject": spec.get("subject"),
                "domain": spec.get("domain"),
                "learning_objective": spec.get("learning_objective"),
                "target_grade_band": spec.get("target_grade_band"),
                "target_stage_in_trajectory": spec.get("target_stage_in_trajectory"),
                "target_learner_state": spec.get("target_learner_state"),
                "target_final_learner_role": spec.get("final_learner_role"),
                "target_interaction_pattern": spec.get("interaction_pattern"),
                "generated_stage_in_trajectory": parsed.get("stage_in_trajectory"),
                "generated_learner_state": parsed.get("learner_state"),
                "generated_final_learner_role": parsed.get("final_learner_role"),
                "generated_interaction_pattern": parsed.get("interaction_pattern"),
                "learner_utterance": parsed.get("learner_utterance"),
                "conversation_setting": parsed.get("conversation_setting"),
                "target_trajectory_mode": spec.get("trajectory_mode"),
                "generated_trajectory_mode": parsed.get("trajectory_mode"),
                "hidden_belief": spec.get("hidden_belief"),  # FIX 4
            })


# Clean training export

def write_clean_training_jsonl(results: List[Dict[str, Any]], path: str = OUTPUT_CLEAN_JSONL) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            if not r["ok"]:
                continue

            parsed = r["parsed_output"]
            spec = r["spec"]

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
                    "target_learner_state_definition": LEARNER_STATE_DEFINITIONS[spec["target_learner_state"]],
                    "final_learner_role_definition": FINAL_LEARNER_ROLE_DEFINITIONS[spec["final_learner_role"]],
                    "interaction_pattern_definition": INTERACTION_PATTERN_DEFINITIONS[spec["interaction_pattern"]],
                    "trajectory_mode": spec["trajectory_mode"],
                    "trajectory_mode_definition": TRAJECTORY_MODE_DEFINITIONS[spec["trajectory_mode"]],
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
            f.write(json.dumps(clean_obj, ensure_ascii=False) + "\n")


# Main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate tutoring conversation snapshots")
    parser.add_argument(
        "--target-clean-samples", type=int, default=50,
        help="Number of clean samples to generate (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: current timestamp, so each run is independent)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to the main output JSONL file. If the file already exists, "
             "generation resumes from it (existing clean samples are counted and "
             "new ones are appended). CSV and clean JSONL are derived from this "
             "path automatically. If omitted, a new timestamped file is created."
    )
    args = parser.parse_args()
    target_clean_samples = args.target_clean_samples
    seed = args.seed if args.seed is not None else int(time.time())

    if args.output:
        out_jsonl = args.output
        out_csv   = args.output.replace(".jsonl", ".csv")
        out_clean = args.output.replace(".jsonl", "_clean.jsonl")
    else:
        out_jsonl = OUTPUT_JSONL
        out_csv   = OUTPUT_CSV
        out_clean = OUTPUT_CLEAN_JSONL

    results = generate_dataset_until_clean(
        target_clean=target_clean_samples,
        seed=seed,
        max_attempts=target_clean_samples * 3,
        max_workers=MAX_WORKERS,
        output_jsonl=out_jsonl,
    )

    write_summary_csv(results, path=out_csv)
    write_clean_training_jsonl(results, path=out_clean)

    ok_count = sum(1 for r in results if r["ok"])
    print(f"Finished. Valid outputs: {ok_count}/{len(results)} attempts")
    print(f"Target clean samples: {target_clean_samples}")
    print(f"JSONL saved to: {out_jsonl}")
    print(f"CSV summary saved to: {out_csv}")
    print(f"Clean JSONL saved to: {out_clean}")
