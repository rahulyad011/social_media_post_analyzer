# Social Media Post Analyzer — Insight Mapper

Maps each research insight to the top 5 most relevant social media posts using a two-stage pipeline:

1. **Vector search** (local, no API) — sentence embeddings via `all-MiniLM-L6-v2` rank all posts by semantic similarity
2. **LLM scoring** (Anthropic or OpenAI) — top candidates are sent to the LLM once per insight for relevance scoring and exact excerpt extraction

---

## Prerequisites

- Python 3.9+
- `insights.xlsx` and `posts.xlsx` in the project directory (see Input Format below)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure `.env`  (secrets only)

Create (or edit) `.env` in the project root:

```env
# Choose your LLM provider: anthropic | openai
LLM_PROVIDER=anthropic

# Anthropic (used when LLM_PROVIDER=anthropic)
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI (used when LLM_PROVIDER=openai)
OPENAI_API_KEY=sk-...
```

Only the key for the chosen provider needs to be set.

### 3. Adjust `config.yaml`  (everything else)

All file paths, column names, tuning parameters, model names, and Excel formatting are in `config.yaml`. Edit it freely — no code changes needed.

---

## Input Format

### `insights.xlsx`

| Column | Description |
|--------|-------------|
| `#` | Insight number |
| `Slide #` | Source slide reference |
| `Priority` | Priority label |
| `Category` | Insight category |
| `Insights` | **The insight text** (used for matching) |
| `Stage 1..6` | AD pathway stage flags (optional) |
| `Patient`, `Clinician`, `Payer`, `Provider`, `Societal` | Stakeholder flags (optional) |

### `posts.xlsx`

| Column | Description |
|--------|-------------|
| `lsc_id` | Unique post ID |
| `post` | **Full post text** (used for matching) |
| `analysis_summary` | Pre-computed summary of the post |
| `author_type` | e.g. Patient, HCP, Caregiver |

---

## Run

```bash
python map_insights_to_posts.py
```

Progress and debug logs are printed to stdout and also written to `run.log`.

---

## Output

**`insights_mapped.xlsx`** — one row per insight, with up to 5 matched posts appended as columns:

| Column | Description |
|--------|-------------|
| *(all insight columns)* | Copied from `insights.xlsx` |
| `Post N ID` | Matched post ID |
| `Post N Author Type` | Author type of matched post |
| `Post N Summary` | Pre-computed summary |
| `Post N Excerpt` | Exact quote from post supporting the insight |
| `Post N Sim Score` | Cosine similarity (0–1) from vector search |
| `Post N AI Score` | Relevance score (0–10) from LLM |

Posts are sorted by AI Score descending. Only posts with AI Score ≥ 7 are included.

---

## Tuning Parameters

Edit the `tuning` section in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_n` | 5 | Max posts per insight in output |
| `top_k_for_llm` | 10 | Candidates sent to LLM per insight |
| `min_similarity` | 0.45 | Minimum cosine similarity to pass to LLM |
| `min_score` | 7 | Minimum LLM score to include in output |
| `llm_sleep_sec` | 1 | Pause between LLM calls (rate-limit buffer) |

---

## Models Used

| Provider | Model |
|----------|-------|
| Anthropic | `claude-haiku-4-5-20251001` |
| OpenAI | `gpt-4o-mini` |

To change models, update `ANTHROPIC_MODEL` or `OPENAI_MODEL` in the config section.
