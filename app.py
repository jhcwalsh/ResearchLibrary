import json
from pathlib import Path

import streamlit as st
import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

from src.research_library.zotero_client import (
    search_papers, get_recent_papers, list_collections,
    get_collection_papers, get_papers_by_tag, get_tags, get_annotations,
    get_fulltext, get_fulltext_batch, extract_pdf_text, PYMUPDF_AVAILABLE,
)
from src.research_library.analysis import build_index, load_cached_index

SUBS_PATH = Path(__file__).parent / "subcategory_suggestions.json"


def load_subcategories() -> dict:
    if SUBS_PATH.exists():
        return json.loads(SUBS_PATH.read_text(encoding="utf-8"))
    return {}

st.set_page_config(page_title="Research Library", page_icon="📚", layout="wide")
st.title("📚 Research Library")

mode = st.sidebar.radio("Mode", ["Search & Ask", "Browse Collections", "Browse Tags", "Recent Papers", "Library Index"])


def format_paper(p):
    authors = ", ".join(p["authors"]) if p["authors"] else "Unknown"
    year = f" ({p['year']})" if p["year"] else ""
    pub = f" — *{p['publication']}*" if p["publication"] else ""
    doi = f" | [DOI](https://doi.org/{p['doi']})" if p["doi"] else ""
    url = f" | [Link]({p['url']})" if p["url"] else ""
    tags = f"\n\n🏷 {', '.join(p['tags'])}" if p["tags"] else ""
    return f"**{p['title']}**{year}\n\n{authors}{pub}{doi}{url}{tags}"


def extract_keywords(query):
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f"Extract 2-4 search keywords from this query for a Zotero academic library search. Return only the keywords, space-separated, nothing else: {query}"
        }]
    )
    return msg.content[0].text.strip()


def ask_claude(question, papers, use_fulltext=False):
    client = anthropic.Anthropic()
    parts = []
    for p in papers:
        ft = p.get("fulltext", {}) if use_fulltext else {}
        if use_fulltext and ft.get("content"):
            trunc = " [truncated]" if ft.get("truncated") else ""
            parts.append(
                f"Title: {p['title']}\nAuthors: {', '.join(p['authors'])}\nYear: {p['year']}\n"
                f"Full text{trunc}:\n{ft['content']}"
            )
        elif p.get("abstract"):
            parts.append(
                f"Title: {p['title']}\nAuthors: {', '.join(p['authors'])}\nYear: {p['year']}\n"
                f"Abstract: {p['abstract']}"
            )

    if not parts:
        parts = [f"- {p['title']} ({', '.join(p['authors'])})" for p in papers]

    context = "\n\n---\n\n".join(parts)
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Based on these papers from my Zotero library, answer this question: {question}\n\nPapers:\n{context}"
        }]
    ) as stream:
        return stream.get_final_text()


def paper_expander(p):
    """Render a single paper expander with metadata, full-text, and annotations."""
    with st.expander(f"{p['title']} ({p['year'] or '?'})"):
        st.markdown(format_paper(p))
        if p["abstract"]:
            st.markdown(f"**Abstract:** {p['abstract']}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Read full text", key=f"ft_{p['key']}"):
                with st.spinner("Fetching full text..."):
                    ft = get_fulltext(p["key"])
                st.session_state[f"ft_result_{p['key']}"] = ft

        with col2:
            if st.button("Get annotations", key=f"ann_{p['key']}"):
                with st.spinner("Fetching annotations..."):
                    annotations = get_annotations(p["key"])
                st.session_state[f"ann_result_{p['key']}"] = annotations

        # Show full text result
        ft = st.session_state.get(f"ft_result_{p['key']}")
        if ft is not None:
            if ft["content"]:
                if ft["truncated"]:
                    st.caption(f"Showing first {len(ft['content']):,} of {ft['total_chars']:,} chars "
                               f"({ft.get('indexed_pages', '?')}/{ft.get('total_pages', '?')} pages indexed)")
                st.text_area("Full text", ft["content"], height=300, key=f"ft_text_{p['key']}")
                paper_q = st.text_input("Ask about this paper", key=f"ft_q_{p['key']}")
                if paper_q:
                    with st.spinner("Thinking..."):
                        answer = ask_claude(paper_q, [{**p, "fulltext": ft}], use_fulltext=True)
                    st.info(answer)
            elif ft["error"] == "not_indexed":
                st.warning("Not indexed in Zotero.")
                if PYMUPDF_AVAILABLE and st.button("Extract from PDF", key=f"pdf_{p['key']}"):
                    with st.spinner("Downloading and extracting PDF..."):
                        ft2 = extract_pdf_text(p["key"])
                    if ft2["content"]:
                        st.text_area("Extracted text", ft2["content"], height=300, key=f"pdf_text_{p['key']}")
                    else:
                        st.error(f"Extraction failed: {ft2['error']}")
            elif ft["error"] == "no_attachment":
                st.warning("No PDF attachment found.")
            else:
                st.error(f"Could not retrieve full text: {ft['error']}")

        # Show annotations result
        annotations = st.session_state.get(f"ann_result_{p['key']}")
        if annotations is not None:
            if annotations:
                for a in annotations:
                    text = a.get("text") or a.get("note") or a.get("comment", "")
                    if text:
                        st.markdown(f"> {text}")
            else:
                st.write("No annotations.")


def _render_paper_list(papers):
    """Render a compact list of papers (title, authors, year, DOI)."""
    for p in papers:
        authors = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors += " et al."
        year = f" ({p['year']})" if p["year"] else ""
        doi_link = f" · [DOI](https://doi.org/{p['doi']})" if p.get("doi") else ""
        st.markdown(f"**{p['title']}**{year}  \n{authors}{doi_link}")
        if p.get("abstract"):
            st.caption(p["abstract"][:200] + ("..." if len(p["abstract"]) > 200 else ""))


# ── Search & Ask ──────────────────────────────────────────────────────────────
if mode == "Search & Ask":
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search your library", placeholder="e.g. regime switching, asset allocation...")
    with col2:
        limit = st.number_input("Results", min_value=5, max_value=100, value=20)

    if query:
        with st.spinner("Searching..."):
            keywords = extract_keywords(query) if len(query.split()) > 3 else query
            if keywords != query:
                st.caption(f"Searching for: *{keywords}*")
            papers = search_papers(keywords, limit=int(limit))

        if not papers:
            st.warning("No papers found.")
        else:
            st.success(f"{len(papers)} papers found")

            st.divider()
            col_q, col_ft = st.columns([3, 1])
            with col_q:
                question = st.text_input("Ask a question about these results",
                                         placeholder="e.g. What methodologies are used?")
            with col_ft:
                use_fulltext = st.checkbox("Use full text", value=False,
                                           help="Fetches full paper text instead of abstracts (slower)")

            if question:
                if use_fulltext:
                    max_ft = st.slider("Max papers to read in full", 1, 10, 5)
                    with st.spinner(f"Fetching full text for up to {max_ft} papers..."):
                        papers_with_text = get_fulltext_batch(papers, max_papers=max_ft, char_limit=3000)
                    available = sum(1 for p in papers_with_text if p.get("fulltext", {}).get("content"))
                    st.caption(f"Full text available for {available} of {min(max_ft, len(papers))} papers")
                else:
                    papers_with_text = papers

                with st.spinner("Thinking..."):
                    answer = ask_claude(question, papers_with_text, use_fulltext=use_fulltext)
                st.info(answer)

            st.divider()
            for p in papers:
                paper_expander(p)

# ── Browse Collections ────────────────────────────────────────────────────────
elif mode == "Browse Collections":
    with st.spinner("Loading collections..."):
        collections = list_collections()

    if not collections:
        st.warning("No collections found.")
    else:
        names = [f"{c['name']} ({c['num_items']})" for c in collections]
        choice = st.selectbox("Collection", names)
        selected = collections[names.index(choice)]

        with st.spinner("Loading papers..."):
            papers = get_collection_papers(selected["name"], limit=100)

        st.success(f"{len(papers)} papers in '{selected['name']}'")
        for p in papers:
            paper_expander(p)

# ── Browse Tags ───────────────────────────────────────────────────────────────
elif mode == "Browse Tags":
    with st.spinner("Loading tags..."):
        tags = get_tags()

    if not tags:
        st.warning("No tags found.")
    else:
        tag = st.selectbox("Tag", tags)
        if tag:
            with st.spinner("Loading papers..."):
                papers = get_papers_by_tag(tag, limit=100)
            st.success(f"{len(papers)} papers tagged '{tag}'")
            for p in papers:
                paper_expander(p)

# ── Recent Papers ─────────────────────────────────────────────────────────────
elif mode == "Recent Papers":
    limit = st.slider("How many?", 5, 50, 20)
    with st.spinner("Loading..."):
        papers = get_recent_papers(limit=limit)

    st.success(f"Showing {len(papers)} most recently added papers")
    for p in papers:
        paper_expander(p)

# ── Library Index ─────────────────────────────────────────────────────────────
elif mode == "Library Index":
    cached = load_cached_index()

    col1, col2 = st.columns([3, 1])
    with col1:
        if cached:
            generated = cached["generated_at"][:19].replace("T", " ")
            st.caption(f"Index generated: {generated} · "
                       f"{cached['unique_papers']} unique papers · "
                       f"{cached['duplicates_removed']} duplicates removed · "
                       f"{len(cached['categories'])} categories")
    with col2:
        regenerate = st.button("Generate / Regenerate Index", type="primary")

    if regenerate or not cached:
        if not cached:
            st.info("No index yet. Generating now — this takes 1–2 minutes...")

        log = st.empty()
        messages = []

        def on_progress(msg):
            messages.append(msg)
            log.text("\n".join(messages[-6:]))

        with st.spinner("Building index..."):
            cached = build_index(progress_callback=on_progress)
        log.empty()
        st.success("Index built and cached.")
        st.rerun()

    if cached:
        subcategories = load_subcategories()
        categories = cached["categories"]

        # ── Level 1: Category dropdown ─────────────────────────────────────
        sorted_cat_names = sorted(categories.keys(),
                                  key=lambda c: len(categories[c]["papers"]), reverse=True)
        cat_options = [f"{c}  ({len(categories[c]['papers'])} papers)" for c in sorted_cat_names]

        selected_cat_label = st.selectbox("Category", cat_options, index=0)
        selected_cat = sorted_cat_names[cat_options.index(selected_cat_label)]

        cat_data = categories[selected_cat]
        cat_papers = cat_data["papers"]
        paper_by_key = {p["key"]: p for p in cat_papers}

        st.markdown(f"*{cat_data['summary']}*")

        # ── Level 2: Subcategory dropdown (if available) ───────────────────
        cat_subs = subcategories.get(selected_cat)
        if cat_subs:
            sub_names_sorted = sorted(cat_subs.keys(), key=lambda s: -len(cat_subs[s]))
            sub_options = ["All subcategories"] + [
                f"{s}  ({len(cat_subs[s])})" for s in sub_names_sorted
            ]
            selected_sub_label = st.selectbox("Subcategory", sub_options, index=0)

            if selected_sub_label == "All subcategories":
                active_papers = cat_papers
                active_subs = cat_subs
            else:
                selected_sub = sub_names_sorted[sub_options.index(selected_sub_label) - 1]
                active_papers = [paper_by_key[k] for k in cat_subs[selected_sub] if k in paper_by_key]
                active_subs = None
        else:
            active_papers = cat_papers
            active_subs = None

        # ── Search within selection ────────────────────────────────────────
        filter_text = st.text_input("Filter papers", placeholder="title, author, keyword...")
        if filter_text:
            q = filter_text.lower()
            active_papers = [p for p in active_papers
                             if q in p["title"].lower()
                             or any(q in a.lower() for a in p.get("authors", []))
                             or q in (p.get("abstract") or "").lower()]

        st.caption(f"{len(active_papers)} papers")
        st.divider()

        # ── Render papers, grouped by subcategory if showing all ───────────
        if active_subs and filter_text == "":
            visible_keys = {p["key"] for p in active_papers}
            for sub_name in sorted(active_subs.keys(), key=lambda s: -len(active_subs[s])):
                sub_papers = [paper_by_key[k] for k in active_subs[sub_name] if k in visible_keys]
                if not sub_papers:
                    continue
                st.markdown(f"#### {sub_name} &nbsp; <small>({len(sub_papers)})</small>",
                            unsafe_allow_html=True)
                _render_paper_list(sub_papers)
                st.divider()
        else:
            _render_paper_list(active_papers)
