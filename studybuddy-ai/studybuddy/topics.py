"""Automatic topic detection.

1. Structure first: slide titles / PDF bookmarks / headings are used when the document has them.
2. AI grouping (optional): the outline of titles is sent to the LLM, which groups consecutive
   sections into named study topics.
3. Fallback when a document has no structure at all: TF-IDF + KMeans clustering.
The student can always rename, merge or delete topics afterwards.
"""
from __future__ import annotations

import math
import re

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

_EXTRA_STOP = {"let", "using", "following", "given", "define", "defined", "called", "say", "said", "note",
               "example", "examples", "exercise", "solution", "thus", "hence", "also", "set", "sets",
               "ordered", "like", "use", "used", "make", "need", "want", "follows"}
STOP_WORDS = list(ENGLISH_STOP_WORDS | _EXTRA_STOP)


def pretty(h: str) -> str:
    """'REFLEXIVE RELATION' -> 'Reflexive Relation' (leave mixed-case headings alone)."""
    h = re.sub(r"\s+", " ", h).strip()
    return h.title() if h.isupper() else h


def make_segments(chunks, target: int = 3000, include_all: bool = False):
    """Split chunks into ordered segments: a new one starts at a heading change, a new document, or
    after `target` characters. Chunks flagged as not-a-topic are skipped unless include_all."""
    segs, cur = [], None
    for i, c in enumerate(chunks):
        if not include_all and not c.in_topics:
            continue
        changed = bool(cur) and bool(c.heading) and (not cur["headings"] or c.heading != cur["headings"][-1])
        if cur is None or c.doc != cur["doc"] or cur["chars"] >= target or changed:
            cur = {"doc": c.doc, "headings": [], "idx": [], "chars": 0, "parent": ""}
            segs.append(cur)
        if c.heading and (not cur["headings"] or cur["headings"][-1] != c.heading):
            cur["headings"].append(c.heading)
        cur["idx"].append(i)
        cur["chars"] += len(c.text)
    return segs


def make_units(chunks, limit: int = 150):
    """What the AI is shown: ALL pages as units - chapters for books (chapter > section bookmarks),
    otherwise slide titles / headings / text segments. Capped so the outline stays small."""
    segs_all = make_segments(chunks, include_all=True)
    units = []
    for doc in dict.fromkeys(c.doc for c in chunks):
        idx = [i for i, c in enumerate(chunks) if c.doc == doc]
        if any(chunks[i].parent for i in idx):
            cur = None
            for i in idx:
                p = chunks[i].parent.strip() or (cur["parent"] if cur else "(front pages)")
                if cur is None or p != cur["parent"]:
                    cur = {"doc": doc, "parent": p, "idx": [], "headings": [], "chars": 0}
                    units.append(cur)
                h = chunks[i].heading.strip()
                if h and h != p and (not cur["headings"] or cur["headings"][-1] != h):
                    cur["headings"].append(h)
                cur["idx"].append(i)
                cur["chars"] += len(chunks[i].text)
        else:
            du = [sg for sg in segs_all if sg["doc"] == doc]
            if len(du) > limit:  # merge neighbours so the outline stays small
                k, merged = math.ceil(len(du) / limit), []
                for j in range(0, len(du), k):
                    grp = du[j:j + k]
                    merged.append({"doc": doc, "parent": "", "idx": [i for g in grp for i in g["idx"]],
                                   "headings": list(dict.fromkeys(h for g in grp for h in g["headings"])),
                                   "chars": sum(g["chars"] for g in grp)})
                du = merged
            units.extend(du)
    return units


def _where(seg, chunks):
    a, b = chunks[seg["idx"][0]].loc, chunks[seg["idx"][-1]].loc
    return a if a == b else f"{a} to {b}"


def _label(seg, chunks):
    if seg.get("parent"):
        secs = "; ".join(pretty(h) for h in seg["headings"][:8])
        text = pretty(seg["parent"]) + (f" - sections: {secs}" if secs else "")
    elif seg["headings"]:
        text = " / ".join(pretty(h) for h in seg["headings"][:4])
    else:
        text = re.sub(r"\s+", " ", chunks[seg["idx"][0]].text)
    return f"{text[:260]} ({_where(seg, chunks)})"


def _doc_title(doc: str) -> str:
    return re.sub(r"\.[A-Za-z0-9]+$", "", doc).replace("_", " ").strip().title()


def _prior(unit, chunks):
    """What the keyword list thinks of a unit: 'keep', 'reference' or 'exclude'."""
    cs = [chunks[i] for i in unit["idx"]]
    if all(not c.searchable for c in cs):
        return "exclude"
    return "reference" if all(not c.in_topics for c in cs) else "keep"


def _ai_doc_topics(llm, units, chunks, doc, target=None):
    """The AI classifies every unit (topic / reference / exclude) and groups and names the topics.
    Chunk flags are only changed once the answer has been validated."""
    book = any(u["parent"] for u in units)
    if book:
        rules = "- This is a book: normally every chapter or appendix becomes its own topic; merge only very short ones."
    else:
        n_heads = len({h for u in units for h in u["headings"]})
        target = target or max(3, min(12, max(len(units) // 2, round(n_heads / 2.5))))
        rules = f"- Aim for about {target} topics (never fewer than {max(2, target - 2)} or more than {target + 2})."
    outline = "\n".join(f"{i}: {_label(u, chunks)}" for i, u in enumerate(units))
    prompt = f"""Below is the ordered outline of a study document ("{doc}"). Each line is a unit id, then its title
(and sections) and where it is. You are preparing it for a student's revision app.

Classify EVERY unit:
1. "exclude": not study content - covers, title and copyright pages, tables of contents, prefaces and
   "how to use this book" notes, acknowledgements, website/companion notices, bibliographies and reference
   lists, indexes, credits, licences, advertisements.
2. "reference": handy to look things up in, but not something to study or be quizzed on - answer keys,
   solutions to exercises, glossaries, formula or table appendices.
3. Everything else is study content: group CONSECUTIVE units into topics, the way a lecturer would divide
   the syllabus.

Topic rules:
{rules}
- Name each topic after the concept(s) it teaches, 2-6 words in Title Case. A chapter may keep its own title
  without the number. Never use generic names such as "Examples", "Exercises" or "Miscellaneous".
- Keep examples, exercises and solutions with the concept they illustrate.
- Every unit id appears exactly once: in one topic, in "exclude", or in "reference".

Return JSON only:
{{"topics": [{{"name": "...", "segments": [0, 1]}}], "exclude": [2], "reference": [3]}}

OUTLINE:
{outline}"""
    data = llm.generate_json(prompt, temperature=0.2, tries=2)
    if isinstance(data, list):
        data = {"topics": data}
    n, decision = len(units), {}
    valid = lambda v: isinstance(v, int) and 0 <= v < n
    for t in data.get("topics") or []:
        name = str(t.get("name", "")).strip()[:60]
        for sid in t.get("segments", []):
            if name and valid(sid):
                decision.setdefault(sid, ("topic", name))
    for sid in data.get("exclude") or []:
        if valid(sid):
            decision.setdefault(sid, ("exclude", ""))
    for sid in data.get("reference") or []:
        if valid(sid):
            decision.setdefault(sid, ("reference", ""))
    names = [v[1] for v in decision.values() if v[0] == "topic"]
    if not names:
        raise ValueError("no topics returned")
    # units the AI did not mention: follow the keyword prior (keep -> join the previous topic)
    final, last = [], names[0]
    for i, u in enumerate(units):
        kind, name = decision.get(i) or {"keep": ("topic", ""), "reference": ("reference", ""),
                                         "exclude": ("exclude", "")}[_prior(u, chunks)]
        if kind == "topic":
            last = name or last
            name = last
        final.append((kind, name))
    total = sum(u["chars"] for u in units) or 1
    if sum(u["chars"] for u, (k, _) in zip(units, final) if k == "exclude") > 0.5 * total:
        raise ValueError("the AI excluded most of the document")
    out = {}
    for u, (kind, name) in zip(units, final):
        for i in u["idx"]:
            chunks[i].in_topics = kind == "topic"
            chunks[i].searchable = kind != "exclude"
        if kind == "topic":
            out.setdefault(name, []).extend(u["idx"])
    if book and target and target > len(out):  # user wants finer topics: split by sections instead
        kept = sorted(i for ids in out.values() for i in ids)
        out = _structure_topics(kept, chunks, target) or out
    return out


def _tokens(h):
    return {w for w in re.findall(r"[a-z]{3,}", h.lower()) if w not in ENGLISH_STOP_WORDS}


def _merge_related(groups):
    """Join neighbouring headings about the same concept (e.g. 'Reflexive Relation', 'Directed Graph of a
    Reflexive Relation', 'Matrix Representation of a Reflexive Relation'). Words found in many headings
    (like 'relation') say nothing about the concept and are ignored."""
    if len(groups) < 3:
        return groups
    df = {}
    for h, _ in groups:
        for w in _tokens(h):
            df[w] = df.get(w, 0) + 1
    common = {w for w, c in df.items() if c >= max(3, 0.35 * len(groups))}
    out, acc = [], set()
    for h, ids in groups:
        toks = _tokens(h) - common
        if out and toks & acc:
            out[-1][1].extend(ids)
            acc |= toks
        else:
            out.append([h, list(ids)])
            acc = set(toks)
    return out


def _structure_topics(idx, chunks, target=None):
    """Topics from the document's own headings. Many headings are merged into ~10 neighbouring groups."""
    if not any(chunks[i].heading for i in idx):
        return None
    if not target:  # books with chapters: one topic per chapter (when there is a sensible number of them)
        chapters = []
        for i in idx:
            p = chunks[i].parent.strip()
            if p and chapters and chapters[-1][0] == p:
                chapters[-1][1].append(i)
            elif p:
                chapters.append([p, [i]])
            elif chapters:
                chapters[-1][1].append(i)
        if 3 <= len(chapters) <= 30:
            return {pretty(h)[:60]: ids for h, ids in chapters}
    groups = []  # consecutive chunks with the same heading
    for i in idx:
        h = chunks[i].heading.strip() or (groups[-1][0] if groups else "Introduction")
        if groups and h == groups[-1][0]:
            groups[-1][1].append(i)
        else:
            groups.append([h, [i]])
    if len(groups) > 1 and groups[0][0] == "Introduction" and len(groups[0][1]) <= 2:
        groups[1][1] = groups[0][1] + groups[1][1]  # a bare title slide joins the first real topic
        groups.pop(0)
    groups = _merge_related(groups)
    want = target or 12
    if len(groups) <= max(want, 15) and not target:
        if len(groups) < 2:
            return None
        return {pretty(h)[:60]: ids for h, ids in groups}
    size = math.ceil(len(groups) / max(1, want))  # merge neighbours (never across chapters), name by headings
    out, window = {}, []

    def flush():
        if window:
            names = list(dict.fromkeys(pretty(h) for h, _ in window))[:2]
            out.setdefault(" / ".join(names)[:60], []).extend(i for _, ids in window for i in ids)
            window.clear()

    for g in groups:
        if window and (len(window) >= size or chunks[g[1][0]].parent != chunks[window[0][1][0]].parent):
            flush()
        window.append(g)
    flush()
    return out


def _cluster_topics(segs, chunks, doc):
    if len(segs) < 3:
        return {_doc_title(doc): [i for s in segs for i in s["idx"]]}
    texts = [" ".join(chunks[i].text for i in s["idx"]) for s in segs]
    try:
        vec = TfidfVectorizer(stop_words=STOP_WORDS, max_features=3000)
        X = vec.fit_transform(texts)
        k = max(2, min(8, len(segs) // 3))
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X)
    except ValueError:
        return {_doc_title(doc): [i for s in segs for i in s["idx"]]}
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for c in range(k):
        members = [j for j, lab in enumerate(km.labels_) if lab == c]
        if not members:
            continue
        name = ", ".join(terms[np.argsort(-km.cluster_centers_[c])[:3]]).title()
        out.setdefault(name, []).extend(i for j in members for i in segs[j]["idx"])
    return out


def _merge(into, new):
    keys = {k.lower(): k for k in into}
    for name, idx in new.items():
        key = keys.get(name.lower(), name)
        into.setdefault(key, []).extend(idx)
        keys[key.lower()] = key


def apply_topics(chunks, topics):
    for c in chunks:
        c.topic = ""
    for name, idx in topics.items():
        for i in idx:
            chunks[i].topic = name


def _heuristic_doc_topics(chunks, doc, target):
    segs = [sg for sg in make_segments(chunks) if sg["doc"] == doc]
    if not segs:
        return {}
    idx = [i for sg in segs for i in sg["idx"]]
    if len(segs) <= 2:
        return {_doc_title(doc): idx}
    return _structure_topics(idx, chunks, target) or _cluster_topics(segs, chunks, doc)


def detect_topics(chunks, llm=None, target=None):
    """Returns (topics, notes). topics: {name: [chunk ids]} in document order.
    With an `llm`, the AI decides what is study content and how to group and name topics (one request per
    document). Without one - or if it fails - headings/bookmarks and a keyword list are used instead.
    `target` = desired number of topics (None = automatic)."""
    units_all = make_units(chunks)
    topics, notes = {}, []
    for doc in dict.fromkeys(c.doc for c in chunks):
        units = [u for u in units_all if u["doc"] == doc]
        found = None
        if llm is not None and len(units) > 2:
            try:
                found = _ai_doc_topics(llm, units, chunks, doc, target)
            except Exception as e:  # LLM down / bad JSON / implausible answer -> fall back
                notes.append(f"AI topic selection failed for '{doc}' ({e}); used the document structure instead.")
        if found is None:
            found = _heuristic_doc_topics(chunks, doc, target)
        _merge(topics, found)
    topics = {k: sorted(set(v)) for k, v in topics.items()}
    apply_topics(chunks, topics)
    return topics, notes
