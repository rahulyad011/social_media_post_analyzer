"""
app.py — Streamlit UI for the Insight ↔ Post Mapper

Pipeline:
  1. Bi-Encoder retrieval  (always)
  2. CrossEncoder reranking (optional, toggle in sidebar)
  3. LLM scoring + excerpt extraction

Run with:
  streamlit run app.py
"""

import os
import re
import json
import time

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Insight ↔ Post Mapper",
    page_icon="🔍",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# Session-state defaults
# ──────────────────────────────────────────────────────────────

_DEFAULTS = {
    "phase":       "setup",   # "setup" | "results"
    "view_step":   0,         # which result stage the user is viewing
    "be_results":  None,      # list[{insight_row, candidates}]
    "ce_results":  None,      # list[{insight_row, candidates}]  or None
    "llm_results": None,      # list[{insight_row, top_posts}]
    "run_cfg":     None,      # dict of runtime config
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ──────────────────────────────────────────────────────────────
# Cached model loaders
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading bi-encoder model…")
def _load_biencoder(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


@st.cache_resource(show_spinner="Loading CrossEncoder model…")
def _load_crossencoder(name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name)


# ──────────────────────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────────────────────

_LLM_PROMPT = """\
You are an expert life sciences analyst reviewing social media posts about Alzheimer's disease (AD).

INSIGHT:
\"\"\"{insight}\"\"\"

For each post below, score its relevance to the insight on a scale of 0–10:
  10 = directly and explicitly supports or illustrates the insight
  7–9 = strongly related, same theme or patient/HCP/payer experience
  4–6 = partially related, tangentially mentions the topic
  1–3 = weak or incidental connection
  0   = completely unrelated

If score >= {min_score}, extract the specific excerpt (1–3 sentences maximum) that most directly
supports the insight. Do NOT paraphrase — copy exact words from the post.
If score < {min_score}, set excerpt to "".

POSTS:
{posts}

Return ONLY a JSON array, no explanation, no markdown:
[{{"post_id": "<id>", "score": <0-10>, "excerpt": "<exact quote or empty string>"}}]"""


def _build_llm_client(provider: str, api_key: str):
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def _call_llm(client, provider: str, model: str, prompt: str) -> str:
    if provider == "openai":
        r = client.chat.completions.create(
            model=model, max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()
    r = client.messages.create(
        model=model, max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip()


def _parse_json(raw: str) -> list:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return []


# ──────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict) -> dict:
    """
    Executes all three stages sequentially.
    Stores intermediate results so the UI can show each stage separately.
    Returns: {be_results, ce_results, llm_results}
    """
    insights_df = cfg["insights_df"]
    posts_df    = cfg["posts_df"]
    base_col    = cfg["base_col"]
    target_col  = cfg["target_col"]
    post_id_col = cfg["post_id_col"]
    ce_enabled  = cfg["ce_enabled"]
    fetch_k     = cfg["top_k_retrieve"] if ce_enabled else cfg["top_k_for_llm"]
    n           = len(insights_df)

    # Number of weighted stages: embed=1 unit, BE=1, CE=1 (opt), LLM=n units
    total_stages = 2 + (1 if ce_enabled else 0) + n
    done = 0

    progress_bar = st.progress(0.0, text="Starting…")
    status_box   = st.empty()

    def _tick(label: str, step: int = 1):
        nonlocal done
        done += step
        progress_bar.progress(min(done / total_stages, 1.0), text=label)
        status_box.info(label)

    # ── Embed ────────────────────────────────────────────────
    _tick("⚙️ Encoding texts with bi-encoder…")
    embed_model  = _load_biencoder(cfg["embed_model"])
    post_embs    = embed_model.encode(
        posts_df[target_col].tolist(), batch_size=64, show_progress_bar=False
    )
    insight_embs = embed_model.encode(
        insights_df[base_col].tolist(), batch_size=64, show_progress_bar=False
    )
    _tick("✅ Encoding complete")

    # ── Stage 1: Bi-Encoder retrieval ────────────────────────
    be_results = []
    for i, (_, row) in enumerate(insights_df.iterrows()):
        sims       = cosine_similarity([insight_embs[i]], post_embs)[0]
        top_idx    = np.argsort(sims)[::-1]
        candidates = []

        for idx in top_idx:
            sim = float(sims[idx])
            if sim < cfg["min_similarity"]:
                break
            post_row = posts_df.iloc[idx]
            candidates.append({
                "post_id":   str(post_row[post_id_col]) if post_id_col in post_row.index else str(idx),
                "post_text": str(post_row[target_col]),
                "similarity": round(sim, 4),
                "_meta":     post_row.to_dict(),   # full row kept for display
            })
            if len(candidates) >= fetch_k:
                break

        be_results.append({"insight_row": row.to_dict(), "candidates": candidates})

    _tick("✅ Bi-encoder retrieval complete")

    # ── Stage 2: CrossEncoder reranking (optional) ───────────
    ce_results = None
    if ce_enabled:
        ce_model   = _load_crossencoder(cfg["ce_model"])
        ce_results = []
        for r in be_results:
            insight_text = str(r["insight_row"][base_col])
            cands        = r["candidates"]
            if cands:
                pairs  = [(insight_text, c["post_text"]) for c in cands]
                scores = ce_model.predict(pairs)
                ranked = sorted(
                    [dict(c, ce_score=round(float(s), 4)) for c, s in zip(cands, scores)],
                    key=lambda x: x["ce_score"], reverse=True,
                )[: cfg["top_k_for_llm"]]
            else:
                ranked = []
            ce_results.append({"insight_row": r["insight_row"], "candidates": ranked})
        _tick("✅ CrossEncoder reranking complete")

    # ── Stage 3: LLM scoring ─────────────────────────────────
    client    = _build_llm_client(cfg["provider"], cfg["api_key"])
    source    = ce_results if ce_enabled else be_results
    llm_results = []

    for i, r in enumerate(source):
        insight_text = str(r["insight_row"][base_col])
        cands        = r["candidates"]
        short_text   = insight_text[:60] + ("…" if len(insight_text) > 60 else "")
        _tick(f"🤖 LLM scoring insight {i + 1}/{n}: {short_text}", step=1)

        if cands:
            posts_block = "\n".join(f"[{c['post_id']}] {c['post_text']}" for c in cands)
            prompt      = _LLM_PROMPT.format(
                insight=insight_text,
                min_score=cfg["min_score"],
                posts=posts_block,
            )
            raw    = _call_llm(client, cfg["provider"], cfg["llm_model"], prompt)
            scored = _parse_json(raw)

            id_to_c   = {c["post_id"]: c for c in cands}
            top_posts = []
            for item in scored:
                pid     = str(item.get("post_id", ""))
                score   = float(item.get("score", 0))
                excerpt = str(item.get("excerpt", "")).strip()
                if score >= cfg["min_score"] and pid in id_to_c:
                    top_posts.append({**id_to_c[pid], "score": score, "excerpt": excerpt})

            top_posts.sort(key=lambda x: x["score"], reverse=True)
            top_posts = top_posts[: cfg["top_n"]]
        else:
            top_posts = []

        llm_results.append({"insight_row": r["insight_row"], "top_posts": top_posts})
        if cands and cfg["llm_sleep_sec"] > 0:
            time.sleep(cfg["llm_sleep_sec"])

    progress_bar.progress(1.0, text="✅ Pipeline complete!")
    time.sleep(0.6)
    progress_bar.empty()
    status_box.empty()

    return {"be_results": be_results, "ce_results": ce_results, "llm_results": llm_results}


# ──────────────────────────────────────────────────────────────
# Sidebar  (rendered on every page)
# ──────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Configuration")

        # ── LLM ──────────────────────────────────────────────
        st.subheader("🤖 LLM")
        provider = st.selectbox("Provider", ["anthropic", "openai"], key="sb_provider")

        env_key = os.getenv(
            "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY", ""
        )
        st.text_input(
            "API Key",
            value=env_key,
            type="password",
            key="sb_api_key",
            help="Leave blank to read from environment variable",
        )
        _default_models = {
            "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            "openai":    os.getenv("OPENAI_MODEL",    "gpt-4o-mini"),
        }
        st.text_input("LLM Model", value=_default_models[provider], key="sb_llm_model")

        # ── Embedding ────────────────────────────────────────
        st.divider()
        st.subheader("📐 Embedding")
        st.text_input("Bi-Encoder Model", value="all-mpnet-base-v2", key="sb_embed_model")

        # ── Thresholds ───────────────────────────────────────
        st.divider()
        st.subheader("🎛️ Thresholds")
        st.slider("Top N  (posts kept in output)", 1, 10, 5, key="sb_top_n")
        st.slider("Top K for LLM  (candidates sent)", 3, 30, 10, key="sb_top_k_for_llm")
        st.slider(
            "Min Similarity  (bi-encoder floor)",
            0.0, 1.0, 0.2, step=0.05, key="sb_min_similarity",
        )
        st.slider("Min LLM Score  (0–10)", 0, 10, 5, key="sb_min_score")
        st.slider("LLM Sleep  (sec between calls)", 0, 5, 1, key="sb_llm_sleep_sec")

        # ── CrossEncoder ──────────────────────────────────────
        st.divider()
        st.subheader("🔄 CrossEncoder")
        ce_on = st.checkbox("Enable CrossEncoder reranking", key="sb_ce_enabled")
        st.text_input(
            "CrossEncoder Model",
            value="cross-encoder/ms-marco-MiniLM-L-6-v2",
            key="sb_ce_model",
            disabled=not ce_on,
        )
        st.slider(
            "Top K to retrieve  (before reranking)",
            10, 200, 50, key="sb_top_k_retrieve",
            disabled=not ce_on,
        )

        # ── Reset ────────────────────────────────────────────
        if st.session_state.phase == "results":
            st.divider()
            if st.button("🔄 Start Over", use_container_width=True):
                for k, v in _DEFAULTS.items():
                    st.session_state[k] = v
                st.rerun()


# ──────────────────────────────────────────────────────────────
# Setup phase
# ──────────────────────────────────────────────────────────────

def render_setup():
    st.title("🔍 Insight ↔ Post Mapper")
    st.markdown(
        "Upload your base and target files, choose which columns to compare, "
        "adjust thresholds in the sidebar, then click **Run Pipeline**."
    )

    # ── File upload ──────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📄 Base File  (Insights)")
        base_file = st.file_uploader(
            "Upload Excel file", type=["xlsx", "xls"], key="fu_base",
            label_visibility="collapsed",
        )
    with col_r:
        st.subheader("📄 Target File  (Posts)")
        target_file = st.file_uploader(
            "Upload Excel file", type=["xlsx", "xls"], key="fu_target",
            label_visibility="collapsed",
        )

    if not base_file or not target_file:
        st.info("📂 Upload both files above to continue.")
        return

    insights_df = pd.read_excel(base_file)
    posts_df    = pd.read_excel(target_file)
    base_cols   = list(insights_df.columns)
    target_cols = list(posts_df.columns)

    # ── Column selection ─────────────────────────────────────
    st.divider()
    st.subheader("🔗 Column Matching")
    st.caption(
        "Select which column from each file should be embedded and compared "
        "for semantic similarity."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        base_col = st.selectbox(
            "Base column  (to embed from insights)",
            base_cols,
            index=0,
            key="sel_base_col",
        )
    with c2:
        target_col = st.selectbox(
            "Target column  (to embed from posts)",
            target_cols,
            index=0,
            key="sel_target_col",
        )
    with c3:
        post_id_col = st.selectbox(
            "Post ID column  (unique identifier)",
            target_cols,
            index=0,
            key="sel_post_id_col",
        )

    # ── File previews ────────────────────────────────────────
    st.divider()
    with st.expander("👁 Preview base file  (first 5 rows)"):
        st.dataframe(insights_df.head(5), use_container_width=True)
    with st.expander("👁 Preview target file  (first 5 rows)"):
        st.dataframe(posts_df.head(5), use_container_width=True)

    # ── Validation ───────────────────────────────────────────
    api_key = st.session_state.get("sb_api_key", "").strip()
    if not api_key:
        st.warning("⚠️ No API key provided in the sidebar.")

    # ── Run ──────────────────────────────────────────────────
    st.divider()
    if st.button("▶  Run Pipeline", type="primary", use_container_width=True):
        cfg = {
            "insights_df":    insights_df,
            "posts_df":       posts_df,
            "base_col":       base_col,
            "target_col":     target_col,
            "post_id_col":    post_id_col,
            "provider":       st.session_state.get("sb_provider", "anthropic"),
            "api_key":        api_key,
            "llm_model":      st.session_state.get("sb_llm_model", "claude-haiku-4-5-20251001"),
            "embed_model":    st.session_state.get("sb_embed_model", "all-mpnet-base-v2"),
            "top_n":          st.session_state.get("sb_top_n", 5),
            "top_k_for_llm":  st.session_state.get("sb_top_k_for_llm", 10),
            "min_similarity": st.session_state.get("sb_min_similarity", 0.2),
            "min_score":      st.session_state.get("sb_min_score", 5),
            "llm_sleep_sec":  st.session_state.get("sb_llm_sleep_sec", 1),
            "ce_enabled":     st.session_state.get("sb_ce_enabled", False),
            "ce_model":       st.session_state.get("sb_ce_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            "top_k_retrieve": st.session_state.get("sb_top_k_retrieve", 50),
        }

        out = run_pipeline(cfg)
        st.session_state.be_results  = out["be_results"]
        st.session_state.ce_results  = out["ce_results"]
        st.session_state.llm_results = out["llm_results"]
        st.session_state.run_cfg     = cfg
        st.session_state.phase       = "results"
        st.session_state.view_step   = 0
        st.rerun()


# ──────────────────────────────────────────────────────────────
# Results phase
# ──────────────────────────────────────────────────────────────

def _candidate_table(candidates: list, show_ce: bool = False) -> pd.DataFrame:
    rows = []
    for rank, c in enumerate(candidates, 1):
        row = {
            "Rank":       rank,
            "Post ID":    c["post_id"],
            "Similarity": c["similarity"],
        }
        if show_ce and "ce_score" in c:
            row["CE Score"] = c["ce_score"]
        preview = c["post_text"]
        row["Text Preview"] = preview[:140] + "…" if len(preview) > 140 else preview
        rows.append(row)
    return pd.DataFrame(rows)


def render_results():
    be_results  = st.session_state.be_results
    ce_results  = st.session_state.ce_results
    llm_results = st.session_state.llm_results
    cfg         = st.session_state.run_cfg
    base_col    = cfg["base_col"]
    ce_on       = cfg["ce_enabled"]

    # Build step list based on what ran
    steps = [("🔍 Bi-Encoder Retrieval", "be")]
    if ce_on:
        steps.append(("🔄 CrossEncoder Reranking", "ce"))
    steps.append(("🤖 LLM Scoring", "llm"))

    vs = st.session_state.view_step
    vs = max(0, min(vs, len(steps) - 1))   # clamp to valid range

    step_label, step_key = steps[vs]

    # ── Header + step indicator ──────────────────────────────
    st.title("Results")
    step_cols = st.columns(len(steps))
    for i, (label, _) in enumerate(steps):
        with step_cols[i]:
            if i < vs:
                st.success(label)
            elif i == vs:
                st.info(f"**{label}**")
            else:
                st.container().markdown(f"<div style='color:grey'>{label}</div>", unsafe_allow_html=True)

    st.divider()

    # ── Insight selector ─────────────────────────────────────
    n_insights = len(be_results)
    insight_labels = [
        f"#{i + 1}  —  {str(r['insight_row'].get(base_col, ''))[:90]}"
        for i, r in enumerate(be_results)
    ]
    sel = st.selectbox(
        "Select insight to inspect",
        options=range(n_insights),
        format_func=lambda i: insight_labels[i],
        key="res_insight_sel",
    )

    # Show full insight text
    insight_text = str(be_results[sel]["insight_row"].get(base_col, ""))
    st.markdown(f"> **Insight:** {insight_text}")
    st.divider()

    # ── Stage-specific content ────────────────────────────────
    if step_key == "be":
        cands = be_results[sel]["candidates"]
        st.markdown(f"**{len(cands)} candidates** retrieved by bi-encoder (similarity ≥ {cfg['min_similarity']})")
        if cands:
            st.dataframe(_candidate_table(cands, show_ce=False), use_container_width=True, hide_index=True)
        else:
            st.warning("No candidates found above the similarity threshold.")

    elif step_key == "ce":
        cands = ce_results[sel]["candidates"]
        st.markdown(f"**{len(cands)} candidates** after CrossEncoder reranking")
        if cands:
            st.dataframe(_candidate_table(cands, show_ce=True), use_container_width=True, hide_index=True)
        else:
            st.warning("No candidates remaining after CrossEncoder reranking.")

    else:  # LLM
        top_posts = llm_results[sel]["top_posts"]
        st.markdown(
            f"**{len(top_posts)} posts** matched  (LLM score ≥ {cfg['min_score']}, "
            f"top {cfg['top_n']} kept)"
        )

        if top_posts:
            for rank, p in enumerate(top_posts, 1):
                ce_badge = f"  |  CE {p['ce_score']:.3f}" if "ce_score" in p else ""
                header   = (
                    f"**#{rank}**  —  Post `{p['post_id']}`"
                    f"  |  🤖 Score **{p['score']}**"
                    f"  |  Sim {p['similarity']}{ce_badge}"
                )
                with st.expander(header, expanded=(rank == 1)):
                    if p.get("excerpt"):
                        st.markdown("**Excerpt**")
                        st.info(p["excerpt"])
                    else:
                        st.caption("No excerpt above threshold.")

                    st.markdown("**Full post text**")
                    st.markdown(p["post_text"])

                    # Show remaining metadata columns
                    meta = p.get("_meta", {})
                    extra = {
                        k: v for k, v in meta.items()
                        if k not in (cfg["target_col"], cfg["post_id_col"])
                        and not k.startswith("_")
                    }
                    if extra:
                        with st.container():
                            st.markdown("**Metadata**")
                            st.dataframe(
                                pd.DataFrame([extra]),
                                use_container_width=True,
                                hide_index=True,
                            )
        else:
            st.warning("No posts met the LLM score threshold for this insight.")

    # ── Navigation ────────────────────────────────────────────
    st.divider()
    nav_l, nav_mid, nav_r = st.columns([1, 4, 1])

    with nav_l:
        if vs > 0:
            if st.button("← Previous", use_container_width=True):
                st.session_state.view_step = vs - 1
                st.rerun()

    with nav_mid:
        st.caption(f"Step {vs + 1} of {len(steps)}: {step_label}")

    with nav_r:
        if vs < len(steps) - 1:
            if st.button("Next →", use_container_width=True, type="primary"):
                st.session_state.view_step = vs + 1
                st.rerun()


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

render_sidebar()

if st.session_state.phase == "results":
    render_results()
else:
    render_setup()
