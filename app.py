import json
import re
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
from src.research_library.llm import HAIKU, SONNET, response_text
from src.research_library.contents import (
    load_topic_summaries, render_citations, short_cite, topic_fingerprint,
)
from src.research_library.links import _extract_year, paper_link
from src.research_library.embeddings import load_embeddings
from src.research_library.linkmap import (
    DEFAULT_MIN_SIMILARITY, DEFAULT_NEIGHBOURS, build_link_graph, nearest_papers,
)
from src.research_library.mapview import render_map_html

def load_subcategories(cached_index: dict) -> dict:
    """Extract {category: {subcategory: [keys]}} from the cached index."""
    return {
        cat_name: cat_data.get("subcategories") or {}
        for cat_name, cat_data in cached_index.get("categories", {}).items()
    }

st.set_page_config(page_title="Research Library", page_icon="📚", layout="wide")
st.title("📚 Research Library")

mode = st.sidebar.radio("Mode", ["Contents", "Paper Map", "Search & Ask", "Browse Collections",
                                 "Browse Tags", "Recent Papers", "Library Index"])


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
        model=HAIKU,
        max_tokens=2000,  # thinking counts toward this; 50 could leave no room for the keywords
        output_config={"effort": "low"},
        messages=[{
            "role": "user",
            "content": f"Extract 2-4 search keywords from this query for a Zotero academic library search. Return only the keywords, space-separated, nothing else: {query}"
        }]
    )
    return response_text(msg).strip()


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
        model=SONNET,
        max_tokens=8000,  # room for thinking as well as the answer
        output_config={"effort": "medium"},
        messages=[{
            "role": "user",
            "content": f"Based on these papers from my Zotero library, answer this question: {question}\n\nPapers:\n{context}"
        }]
    ) as stream:
        return response_text(stream.get_final_message())


def md_safe(text: str) -> str:
    """Stop titles and summaries tripping Streamlit markdown: $ starts LaTeX, [ ] break links."""
    return (text or "").replace("$", r"\$")


def paper_line(p: dict) -> str:
    # Zotero keeps italics as HTML in some titles, e.g. "Returns: <i>58 Years</i>"
    title = re.sub(r"<[^>]+>", "", p["title"])
    title = md_safe(title).replace("[", "(").replace("]", ")")
    return f"- [{title}]({paper_link(p)}) · {short_cite(p)}"


def by_year_desc(papers: list[dict]) -> list[dict]:
    return sorted(papers, key=lambda p: _extract_year(p.get("year") or "") or "0", reverse=True)


def render_topic(name: str, cat: dict, summary: dict | None, papers_by_key: dict):
    """One topic on the Contents page: cited overview, papers to start with, subtopics."""
    papers = cat["papers"]
    subs = cat.get("subcategories") or {}
    st.subheader(name)
    st.caption(f"{len(papers)} papers · {len(subs)} subtopics")

    if summary:
        if summary.get("fingerprint") != topic_fingerprint(cat):
            st.caption("This summary predates the topic's latest papers. "
                       "Run `research-admin contents` to refresh it.")
        st.markdown(render_citations(md_safe(summary["overview"]), papers_by_key))
        key_papers = [papers_by_key[k] for k in summary.get("key_papers", []) if k in papers_by_key]
        if key_papers:
            st.markdown("**Start with**")
            st.markdown("\n".join(paper_line(p) for p in key_papers))
    else:
        st.markdown(f"*{md_safe(cat['summary'])}*")

    st.markdown("#### Subtopics")
    by_key = {p["key"]: p for p in papers}
    grouped = set()
    notes = (summary or {}).get("subtopics", {})
    for sub in sorted(subs, key=lambda s: -len(subs[s])):
        sub_papers = [by_key[k] for k in subs[sub] if k in by_key]
        grouped.update(subs[sub])
        with st.expander(f"{sub} ({len(sub_papers)})"):
            if notes.get(sub):
                st.markdown(render_citations(md_safe(notes[sub]), papers_by_key))
            st.markdown("\n".join(paper_line(p) for p in by_year_desc(sub_papers)))
    rest = [p for p in papers if p["key"] not in grouped]
    if rest:
        with st.expander(f"Other papers ({len(rest)})"):
            st.markdown("\n".join(paper_line(p) for p in by_year_desc(rest)))


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
if mode == "Contents":
    cached = load_cached_index()
    if not cached:
        st.info("No library index yet. Build one under Library Index, or run `research-admin refresh`.")
    else:
        categories = cached["categories"]
        summaries = load_topic_summaries()
        all_papers = {p["key"]: p for c in categories.values() for p in c["papers"]}
        topics = sorted(categories, key=lambda c: -len(categories[c]["papers"]))

        written = [s["generated_at"] for s in summaries.values() if s.get("generated_at")]
        st.caption(f"{cached['unique_papers']} papers · {len(topics)} topic areas"
                   + (f" · summaries written {max(written)[:10]}" if written else ""))
        if not summaries:
            st.info("Topic summaries haven't been written yet. Run `research-admin contents`. "
                    "Showing the short index summaries for now.")

        toc_col, body_col = st.columns([1, 3], gap="large")
        with toc_col:
            topic_filter = st.text_input("Filter topics", placeholder="e.g. private equity")
            shown = [t for t in topics if topic_filter.lower() in t.lower()]
            topic = st.radio("Topic areas", shown, key="contents_topic",
                             format_func=lambda t: f"{t} ({len(categories[t]['papers'])})") if shown else None
            if not shown:
                st.caption("No topic matches that filter.")
        with body_col:
            if topic:
                render_topic(topic, categories[topic], summaries.get(topic), all_papers)


# ── Paper Map ─────────────────────────────────────────────────────────────────
elif mode == "Paper Map":
    cached = load_cached_index()
    emb = load_embeddings()
    if not cached or not emb:
        st.info("The map needs the library index and embeddings. "
                "Run `research-admin refresh` then `research-admin embed`.")
    else:
        categories = cached["categories"]
        topics = sorted(categories, key=lambda c: -len(categories[c]["papers"]))
        all_papers = {p["key"]: p for c in categories.values() for p in c["papers"]}

        c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
        scope = c1.selectbox("Scope", ["Whole library"] + topics,
                             format_func=lambda t: t if t == "Whole library"
                             else f"{t} ({len(categories[t]['papers'])})")
        neighbours = c2.slider("Links per paper", 1, 8, DEFAULT_NEIGHBOURS)
        floor = c3.slider("Similarity floor", 0.30, 0.90, DEFAULT_MIN_SIMILARITY, 0.01)
        highlight = c4.text_input("Highlight papers matching", placeholder="title or author")

        graph = build_link_graph(cached, emb, topic=None if scope == "Whole library" else scope,
                                 neighbours=neighbours, min_similarity=floor)
        s = graph["stats"]
        st.caption(f"{s['papers']} papers · {s['links']} links · {s['unlinked']} with no link at this floor. "
                   "A line joins two papers whose title and abstract say similar things "
                   f"(embedding similarity ≥ {floor:.2f}); these are not citations. "
                   "Hover a dot for its links, click to open the paper, drag to rearrange, scroll to zoom.")
        if graph["nodes"]:
            st.components.v1.html(render_map_html(graph, highlight, height=720), height=740)

        st.markdown("#### Nearest papers")
        keys = sorted(all_papers, key=lambda k: all_papers[k]["title"].lower())
        picked = st.selectbox("Pick a paper", keys, index=None, placeholder="Type to search the library",
                              format_func=lambda k: f"{re.sub(r'<[^>]+>', '', all_papers[k]['title'])}"
                                                    f" · {short_cite(all_papers[k])}")
        if picked:
            lines = []
            for near in nearest_papers(picked, cached, emb, n=12):
                lines.append(f"{paper_line(near['paper'])} · {near['score']:.2f} · *{near['topic']}*")
            st.markdown(f"Closest to [{re.sub(r'<[^>]+>', '', all_papers[picked]['title'])}]"
                        f"({paper_link(all_papers[picked])}), best first:")
            st.markdown("\n".join(lines))


elif mode == "Search & Ask":
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
                    tried = sum(1 for p in papers_with_text if "fulltext" in p)
                    st.caption(f"Full text found for {available} of the {max_ft} requested "
                               f"({tried} papers checked)")
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
        subcategories = load_subcategories(cached)
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
