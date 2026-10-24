# Paper Map: a graphical view that links the research papers

*23 October 2026*

## Intent

James asked for "a graphical interface that links the research papers", and the session
notes named the cross-paper linkage map as the next project. The library already has a list
view (Library Index) and a narrative view (Contents). This adds a map: every paper is a dot,
and papers that say similar things are joined by a line, so a reader can see how the library
hangs together, spot clusters that cross topic boundaries, and move from one paper to its
nearest relatives.

Also in scope: a Windows `.bat` launcher that can be copied to the desktop and starts the
Streamlit app.

## Goals

- A "Paper Map" page in the Streamlit app showing a force-directed graph of the papers.
- Links come from the existing voyage-finance-2 embeddings (`.cache/embeddings.npz`): each
  paper is joined to its *k* most similar papers above a similarity floor. No new model calls,
  no network access, no new cached files.
- Nodes are coloured by topic area and sized by how many links they have. Hover shows title
  and citation; click opens the paper (DOI, URL, else Google Scholar) in a new tab.
- Controls: scope (whole library or one topic), links per paper, similarity floor, a text box
  that highlights matching papers on the map.
- A "Pick a paper" panel under the map listing that paper's nearest neighbours as links with
  their similarity, which also works without the graph for anyone who prefers a list.
- A launcher `Research Library.bat` in the repo root, written with the absolute repo path so it
  works from the desktop.

## Non-goals

- Citation links (who cites whom). Zotero does not hold that, and fetching it is a separate
  project. The map's edges mean "similar content", and the page says so.
- Serving the map over MCP. The map data can be exposed to Claude Desktop later.
- Persisting layouts or user edits.

## Anti-goals

- A hairball. The defaults (4 links per paper, floor around 0.55) must give a readable picture
  of the 733-paper library on a laptop screen; the controls let James thin or thicken it.
- A new heavyweight dependency. The graph is drawn with d3 loaded from cdnjs inside a
  Streamlit HTML component, so the Python side stays numpy-only.

## How it is built (my calls)

- `src/research_library/linkmap.py`: `build_link_graph(index, embeddings, topic, neighbours,
  min_similarity)` returns `{nodes, edges, stats}`; `nearest_papers(key, index, embeddings, n)`
  returns the ranked neighbours for one paper. Pure functions over the two caches, unit-tested
  with small synthetic vectors.
- `app.py`: new sidebar mode "Paper Map" after "Contents". The graph is an inline HTML page
  rendered with `st.components.v1.html`, data passed as JSON. Node payload is key, title,
  citation, topic, link and degree only, so the whole library is well under a megabyte.
- Edges are undirected and deduplicated (A's neighbour B and B's neighbour A is one line).
- Similarity floor default 0.55: the discovery project found related with-abstract papers
  score 0.49-0.61 against each other, so 0.55 keeps the strong pairs and drops noise.

## Left to the builder

- Exact d3 force parameters and colours.
- Whether to draw labels on the map at all, or only on hover and highlight.
