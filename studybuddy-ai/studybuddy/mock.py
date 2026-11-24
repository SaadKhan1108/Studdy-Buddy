"""Offline demo mode: a stand-in for Gemini that needs no API key and uses no quota.

It does NOT write AI answers - it extracts passages and builds fill-in-the-blank questions
from the student's own text, so the whole app (upload, topics, Q&A, notes, quiz, weak spots,
exam plan) can be tested without any requests.
"""
from __future__ import annotations

import random
import re

_WORD = re.compile(r"[A-Za-z][A-Za-z-]{5,}")


def _sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if len(p.strip()) > 40]


class MockLLM:
    model = "demo-mode"

    def generate(self, prompt, system=None, **kw):
        if "single word: ready" in prompt:
            return "ready"
        if "Condense this study material" in prompt:
            return prompt.split(":\n\n", 1)[-1][:3000]
        if "study notes in Markdown" in prompt:
            return self._notes(prompt)
        if "EXCERPTS:" in prompt:
            return self._answer(prompt)
        return "Demo mode response."

    def generate_json(self, prompt, system=None, **kw):
        if "multiple-choice questions" in prompt:
            return self._quiz(prompt)
        if "ordered outline" in prompt:
            return self._topics(prompt)
        return {}

    # ------------------------------------------------------------------
    def _answer(self, prompt):
        body = prompt.split("EXCERPTS:\n", 1)[1].split("\n\nEARLIER CONVERSATION", 1)[0]
        blocks = [b for b in re.split(r"\n\n(?=\[\d+\] )", body) if b.strip()]
        first = blocks[0].split("\n", 1)[-1][:600]
        more = f"\n\nAlso relevant: [{2}]" if len(blocks) > 1 else ""
        return (f"**Demo mode (no AI).** The most relevant passage in your material says:\n\n> {first} [1]{more}")

    def _notes(self, prompt):
        topic = (re.search(r'for the topic "(.*?)"', prompt) or [None, "Topic"])[1]
        material = prompt.split("MATERIAL:\n", 1)[-1]
        points = _sentences(re.sub(r"\([^)]*\)\s*", "", material))[:12]
        bullets = "\n".join(f"- {p}" for p in points)
        return f"## Key points: {topic}\n\n{bullets}\n\n_Demo mode: extracted from your text, not written by AI._"

    def _topics(self, prompt):
        from .ingest import _kind
        lines = re.findall(r"^(\d+): (.*)$", prompt.split("OUTLINE:", 1)[-1], re.M)
        topics, exclude, reference, pending = [], [], [], []

        def flush():
            for k in range(0, len(pending), 2):
                grp = pending[k:k + 2]
                name = " ".join(grp[0][1].split(" (")[0].split(" - sections")[0].split()[:4]).strip(" .:-")
                topics.append({"name": name or f"Part {len(topics) + 1}", "segments": [i for i, _ in grp]})
            pending.clear()

        for i, label in lines:
            kind = _kind(label.split(" (p.")[0].split(" - sections")[0])
            if kind == "drop":
                exclude.append(int(i))
            elif kind == "notopic":
                reference.append(int(i))
            else:
                pending.append((int(i), label))
        flush()
        return {"topics": topics, "exclude": exclude, "reference": reference}

    def _quiz(self, prompt):
        n = int(re.search(r"Create (\d+) multiple-choice", prompt).group(1))
        material = prompt.split("TOPICS AND MATERIAL:", 1)[-1]
        per_topic, pool = {}, set()
        for block in material.split("### TOPIC: ")[1:]:
            name, _, rest = block.partition("\n")
            for line in rest.splitlines():
                m = re.match(r"^\((.*?)\)\s*(.*)", line)
                src, text = (m.group(1), m.group(2)) if m else ("", line)
                for s in _sentences(text):
                    words = _WORD.findall(s)
                    pool.update(w.lower() for w in words)
                    if words:
                        per_topic.setdefault(name.strip(), []).append((src, s, max(words, key=len)))
        questions, queues = [], [(t, list(v)) for t, v in per_topic.items()]
        while len(questions) < n and any(q for _, q in queues):
            for topic, items in queues:
                if not items or len(questions) >= n:
                    continue
                src, sentence, answer = items.pop(random.randrange(len(items)))
                distractors = random.sample([w for w in pool if w != answer.lower()], k=min(3, len(pool) - 1))
                if len(distractors) < 3:
                    continue
                questions.append({
                    "topic": topic, "question": "Fill in the blank: " + sentence.replace(answer, "_____", 1),
                    "options": [answer.lower()] + distractors, "answer_index": 0,
                    "explanation": f"The original sentence uses '{answer.lower()}'.", "source": src})
        return {"questions": questions}
