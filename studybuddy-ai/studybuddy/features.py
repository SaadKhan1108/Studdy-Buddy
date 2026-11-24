"""The AI features: Q&A with citations, notes, quizzes."""
from __future__ import annotations

import random

import numpy as np

from .mathfmt import MATH_RULE

TUTOR_SYSTEM = (
    "You are StudyBuddy, a patient and accurate university tutor. Answer ONLY from the "
    "numbered excerpts of the student's own study material. Cite the excerpts you use like "
    "[1] or [2][3] right after the claim. If the excerpts do not contain the answer, say "
    "\"I couldn't find this in your material.\" and, if helpful, add a short hint clearly "
    "labelled \"Outside your material:\". Never invent facts or page numbers. "
    + MATH_RULE
)

MODES = {
    "Clear & concise": "Answer clearly and concisely.",
    "Explain it simply": ("The student finds this hard. Use plain language, one concrete "
                          "everyday analogy, and build up step by step. Define any jargon."),
    "Go deeper": ("Give a thorough, rigorous explanation including the reasoning behind each "
                  "step, edge cases, and a short worked example if relevant."),
}


def _where(c) -> str:
    return f"{c.doc}, {c.loc}" + (f", '{c.heading}'" if c.heading else "")


def answer_question(llm, index, question, history, mode, allowed=None, k=6):
    """Returns (answer_text, chunks_used)."""
    query = question
    if len(question.split()) < 8 and history:  # short follow-up: add earlier question for search
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        query = f"{last_user} {question}"
    used = index.search(query, k=k, allowed=allowed)
    if not used:
        return ("I couldn't find this in your material. Try using words from your slides or "
                "book, or check the topic filter."), []
    excerpts = "\n\n".join(f"[{n}] ({_where(c)})\n{c.text}" for n, c in enumerate(used, 1))
    convo = "\n".join(
        f"{'Student' if m['role'] == 'user' else 'Tutor'}: {m['content'][:600]}"
        for m in history[-6:]
    )
    prompt = (f"{MODES[mode]}\n\nEXCERPTS:\n{excerpts}\n\nEARLIER CONVERSATION:\n{convo or '(none)'}\n\n"
              f"STUDENT QUESTION: {question}")
    return llm.generate(prompt, system=TUTOR_SYSTEM, temperature=0.3), used


# ------------------------------------------------------------------ notes

def _pack(blocks, limit):
    parts, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) > limit:
            parts.append(cur)
            cur = ""
        cur += b + "\n\n"
    if cur:
        parts.append(cur)
    return parts


def sample_chunks(chunks, budget):
    """Pick chunks spread evenly through the list so they fit in `budget` characters."""
    total = sum(len(c.text) for c in chunks)
    if total <= budget or len(chunks) <= 1:
        return list(chunks)
    k = max(1, int(len(chunks) * budget / total))
    idx = sorted(set(np.linspace(0, len(chunks) - 1, k).astype(int)))
    return [chunks[i] for i in idx]


NOTES_PROMPT = """Write concise, well-organised study notes in Markdown for the topic "{topic}",
using ONLY the material below. Keep ALL important information: definitions, formulas,
key terms (in **bold**), steps/algorithms, examples, exceptions and anything likely to
be examined. Do not add outside facts. Use these sections (skip any that don't apply):
## Overview, ## Key concepts, ## Formulas & rules, ## Worked examples,
## Common mistakes & exam tips, ## Quick recap (5 bullet points).
Mention the source location in brackets, e.g. (p. 12), where useful.
""" + MATH_RULE + """

MATERIAL:
{material}"""


def summarize_topic(llm, topic, chunks, max_parts=None):
    max_parts = max_parts or getattr(llm, "notes_parts", 2)
    blocks = [f"({_where(c)}) {c.text}" for c in chunks]
    parts = _pack(blocks, getattr(llm, "notes_chars", 120000))  # big window = usually ONE request
    if len(parts) > max_parts:  # huge topic: sample evenly
        parts = [parts[i] for i in np.linspace(0, len(parts) - 1, max_parts).astype(int)]
    if len(parts) > 1:  # map step: condense each part, then write the final notes
        partial = [
            llm.generate("Condense this study material into detailed bullet points, keeping every "
                         "definition, formula, example and key fact, and the (source location):\n\n" + p)
            for p in parts
        ]
        material = "\n\n".join(partial)
    else:
        material = parts[0]
    return llm.generate(NOTES_PROMPT.format(topic=topic, material=material), temperature=0.2)


# ------------------------------------------------------------------- quiz

QUIZ_PROMPT = """Create {n} multiple-choice questions for a university student using ONLY the material
below. Difficulty: {difficulty}. Spread the questions roughly evenly across the topics.

Rules:
- Exactly 4 options per question, exactly one clearly correct; distractors must be plausible
  (typical misconceptions), never "all of the above" / "none of the above".
- Test understanding, not trivia. Do not put letters (A/B/C) at the start of options.
- "topic" must be copied exactly from the topic names given.
- "explanation": 1-2 sentences saying why the answer is right.
- "source": the document and location where the answer is found, from the material labels.
- Math in any field: use $...$ for formulas (never \\( \\) or \\[ \\]); inside JSON strings escape every backslash as \\\\.

Return JSON only:
{{"questions": [{{"topic": "...", "question": "...", "options": ["...", "...", "...", "..."],
"answer_index": 0, "explanation": "...", "source": "..."}}]}}

TOPICS AND MATERIAL:
{material}"""


def generate_quiz(llm, topics, n=5, difficulty="Mixed"):
    """topics: {topic name: [Chunk, ...]}. Returns a list of validated question dicts."""
    names = list(topics)
    per_topic = max(1200, getattr(llm, "quiz_chars", 30000) // max(1, len(names)))
    material = ""
    for name in names:
        picked = sample_chunks(topics[name], per_topic)
        material += f"\n### TOPIC: {name}\n" + "\n".join(f"({_where(c)}) {c.text}" for c in picked) + "\n"
    data = llm.generate_json(
        QUIZ_PROMPT.format(n=n, difficulty=difficulty, material=material), temperature=0.5)
    items = data.get("questions") if isinstance(data, dict) else data
    lookup = {t.lower(): t for t in names}
    out = []
    for q in items or []:
        try:
            opts = [str(o).strip() for o in q["options"]]
            ai = int(q["answer_index"])
            if len(opts) != 4 or len(set(opts)) != 4 or not 0 <= ai < 4 or not str(q["question"]).strip():
                continue
            correct = opts[ai]
            random.shuffle(opts)  # models tend to favour certain positions
            out.append({
                "topic": lookup.get(str(q.get("topic", "")).strip().lower(), names[0]),
                "question": str(q["question"]).strip(),
                "options": opts,
                "answer_index": opts.index(correct),
                "explanation": str(q.get("explanation", "")).strip(),
                "source": str(q.get("source", "")).strip(),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out[:n]
