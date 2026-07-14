# TutorLens

**An adaptive framework for evaluating AI tutors in active learning settings.**

TutorLens generates diverse tutoring conversation snapshots, pairs each with a per-conversation rubric that diagnoses the learner's state and specifies the ideal next move, and scores candidate tutor responses with an LLM judge validated against teacher annotations. Unlike prior benchmarks that apply fixed dimensions uniformly, TutorLens adapts its evaluation criteria to each specific dialogue context, targeting open-ended knowledge acquisition settings where no single correct answer exists.

---

## Repository structure

```
├── dataset_generation/
│   ├── conversation_generation_Gemini_agents.py   # Multi-agent pipeline for generating tutoring snapshots
│   └── rebuild_clean_jsonl.py                     # Post-processing and filtering of generated snapshots
│
├── rubric_generation/
│   └── generate_rubrics.py                        # Per-conversation rubric generation via Gemini batch API
│
└── scoring_pipeline/
    ├── get_model_responses.py                     # Collects tutor responses from evaluated models
    └── score_responses.py                         # Scores responses against rubrics using an LLM judge
```

---

## Dataset

The released dataset contains 1,000 tutoring snapshots instantiating the framework, along with model responses from five LLMs under two prompting conditions.

| File | Description |
|---|---|
| `generated_snapshots_grounding_Gemini_agents_final_1000.jsonl` | 1,000 tutoring conversation snapshots |
| `rubrics_1000_v3_final.jsonl` | Per-conversation rubrics for all 1,000 snapshots |
| `responses_<model>_final.jsonl` | Rubric-prompted model responses (main condition) |
| `responses_<model>_baseline_final.jsonl` | Baseline model responses (no pedagogical prompting) |

Models covered: `gemini-3.1-pro-preview`, `gpt-5.5`, `meta-llama/Llama-3.3-70B-Instruct`, `google/gemma-4-31B-it`, `Qwen/Qwen3.6-35B-A3B`.

---

## Reproducing the pipeline

### Requirements

```bash
pip install google-genai python-dotenv openai
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```

### Step 1 — Generate conversation snapshots

```bash
python dataset_generation/conversation_generation_Gemini_agents.py
```

Outputs a JSONL file of tutoring snapshots. Each record contains the conversation history, learner utterance, latent state summary, and schema metadata.

### Step 2 — Generate rubrics

```bash
python rubric_generation/generate_rubrics.py \
    --conversations path/to/snapshots.jsonl \
    --output-dir rubric_output/ \
    --generate
```

Pairs each snapshot with a rubric containing a learner state analysis, ideal next move, and 4–8 weighted criteria items.

### Step 3 — Collect model responses

```bash
python scoring_pipeline/get_model_responses.py \
    --rubrics path/to/rubrics.jsonl \
    --conversations path/to/snapshots.jsonl \
    --output-dir responses/ \
    --gemini-models gemini-3.1-pro-preview \
    --openai-models gpt-5.5
```

### Step 4 — Score responses

```bash
python scoring_pipeline/score_responses.py \
    --responses responses/responses_*.jsonl \
    --rubrics path/to/rubrics.jsonl \
    --conversations path/to/snapshots.jsonl \
    --output-dir scores/ \
    --judge-model gemini-3.1-pro-preview
```

Outputs one JSONL per response file with fields: `id`, `model`, `response`, `rating`, `status`.

---