# Social Media Post Analyzer — Insight Mapper

Maps each research insight to the top-N most relevant social media posts using a three-stage pipeline:

1. **Bi-Encoder retrieval** (local, no API) — sentence embeddings rank all posts by semantic similarity
2. **CrossEncoder reranking** (optional, local) — re-scores top candidates as insight+post pairs
3. **LLM scoring** (Anthropic or OpenAI) — extracts exact excerpts and scores final relevance

Two ways to run: a **CLI script** for batch processing or a **Streamlit UI** for interactive exploration.

---

## Prerequisites

- Python 3.9+
- Input Excel files (see [Input Format](#input-format) below)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure `.env`  (secrets only)

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | Yes | `anthropic` | Which LLM to use: `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | If `LLM_PROVIDER=anthropic` | — | Your Anthropic API key |
| `ANTHROPIC_MODEL` | No | `claude-haiku-4-5-20251001` | Anthropic model to use |
| `OPENAI_API_KEY` | If `LLM_PROVIDER=openai` | — | Your OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model to use |

> Never commit `.env` to version control. Model env vars take priority over `config.yaml` values.

### 3. Adjust `config.yaml`  (everything else)

All file paths, column names, tuning parameters, model names, and Excel formatting live in `config.yaml`. Edit it freely — no code changes needed.

---

## Running

### Option A — Streamlit UI  *(recommended for interactive use)*

```bash
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`.

**Workflow:**
1. Upload your base (insights) and target (posts) Excel files via the file pickers
2. Select which column from each file to use for semantic comparison
3. Pick the post ID column
4. Adjust thresholds and model settings in the left sidebar
5. Click **▶ Run Pipeline**
6. Browse results stage by stage:
   - **Step 1 — Bi-Encoder Retrieval:** candidates ranked by cosine similarity
   - **Step 2 — CrossEncoder Reranking** *(if enabled)*: candidates re-scored as pairs
   - **Step 3 — LLM Scoring:** final matched posts with AI scores and exact excerpts
7. Use **Next →** / **← Previous** to navigate between stages
8. Click **🔄 Start Over** in the sidebar to run a new batch

> API keys can be entered in the sidebar or pre-set via `.env` (auto-loaded on startup).

---

### Option B — CLI script  *(recommended for batch / automated runs)*

Place your files in `data/` (or update paths in `config.yaml`), then:

```bash
python map_insights_to_posts.py
```

Progress and debug logs are printed to stdout and written to `run.log`.

Output: `data/insights_mapped.xlsx`

---

## Input Format

### Base file — `insights.xlsx`

| Column | Description |
|--------|-------------|
| `#` | Insight number |
| `Slide #` | Source slide reference |
| `Priority` | Priority label |
| `Category` | Insight category |
| `Insights` | **The insight text** — used for semantic matching |
| `Stage 1..6` | AD pathway stage flags (optional) |
| `Patient`, `Clinician`, `Payer`, `Provider`, `Societal` | Stakeholder flags (optional) |

### Target file — `posts.xlsx`

| Column | Description |
|--------|-------------|
| `lsc_id` | Unique post ID |
| `post` | **Full post text** — used for semantic matching |
| `analysis_summary` | Pre-computed summary of the post |
| `author_type` | e.g. Patient, HCP, Caregiver |

---

## CLI Output

**`data/insights_mapped.xlsx`** — one row per insight, with up to N matched posts appended as columns:

| Column | Description |
|--------|-------------|
| *(all insight columns)* | Copied from the base file |
| `Post N ID` | Matched post ID |
| `Post N Author Type` | Author type of matched post |
| `Post N Summary` | Pre-computed summary |
| `Post N Excerpt` | Exact quote from post supporting the insight |
| `Post N Sim Score` | Cosine similarity (0–1) from bi-encoder |
| `Post N AI Score` | Relevance score (0–10) from LLM |

Posts are sorted by AI Score descending.

---

## Tuning Parameters

Edit the `tuning` section in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_n` | 5 | Max posts per insight kept in output |
| `top_k_for_llm` | 10 | Candidates sent to LLM per insight |
| `min_similarity` | 0.2 | Cosine similarity floor (bi-encoder stage) |
| `min_score` | 5 | Minimum LLM score to include in output |
| `llm_sleep_sec` | 1 | Pause between LLM calls (rate-limit buffer) |

---

## CrossEncoder

To enable the optional reranking stage, set in `config.yaml`:

```yaml
cross_encoder:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k_retrieve: 50   # bi-encoder fetches this many before reranking
```

In the Streamlit UI, toggle **Enable CrossEncoder reranking** in the sidebar.

---

## Models

| Component | Default model | Config key |
|-----------|--------------|------------|
| Bi-Encoder | `all-mpnet-base-v2` | `models.embedding` |
| CrossEncoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `cross_encoder.model` |
| Anthropic LLM | `claude-haiku-4-5-20251001` | `models.anthropic` |
| OpenAI LLM | `gpt-4o-mini` | `models.openai` |

All model names can be changed in `config.yaml` without touching the code.
