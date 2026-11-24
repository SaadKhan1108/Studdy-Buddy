"""Offline tests (no API key needed) - run with:  python -m pytest -q"""
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

from studybuddy import store
from studybuddy.features import answer_question, generate_quiz, summarize_topic
from studybuddy.ingest import build_chunks
from studybuddy.retrieval import Index
from studybuddy.topics import detect_topics

SAMPLE = Path(__file__).parent.parent / "samples" / "intro_to_machine_learning.md"


class FakeLLM:
    """Stands in for Gemini so the pipeline can be tested offline."""
    def __init__(self, json_reply=None):
        self.json_reply, self.prompts = json_reply, []

    def generate(self, prompt, system=None, **kw):
        self.prompts.append(prompt)
        return "fake answer [1]"

    def generate_json(self, prompt, system=None, **kw):
        self.prompts.append(prompt)
        return self.json_reply


@pytest.fixture
def chunks():
    return build_chunks([(SAMPLE.name, SAMPLE.read_bytes())])


def test_chunks_keep_headings(chunks):
    assert {c.heading for c in chunks} >= {"Supervised Learning", "Gradient Descent"}


def test_retrieval_finds_right_section(chunks):
    top = Index(chunks).search("what is the learning rate in gradient descent?", k=2)
    assert top and top[0].heading == "Gradient Descent"


def test_topics_fallback_without_llm(chunks):
    topics, _ = detect_topics(chunks, None)
    assert topics and sum(len(v) for v in topics.values()) == len(chunks)


def test_topics_with_llm(chunks):
    reply = {"topics": [{"name": "Learning types", "segments": [0, 1]}, {"name": "Training", "segments": [2, 3]}]}
    topics, _ = detect_topics(chunks, FakeLLM(reply))
    assert set(topics) <= {"Learning types", "Training"} and topics
    assert all(c.topic for c in chunks)


def test_answer_without_match_does_not_call_llm(chunks):
    llm = FakeLLM()
    text, used = answer_question(llm, Index(chunks), "zzzz qqqq", [], "Clear & concise")
    assert used == [] and "couldn't find" in text and not llm.prompts


def test_quiz_validation_and_shuffle(chunks):
    topics, _ = detect_topics(chunks, None)
    name = next(iter(topics))
    good = {"topic": name, "question": "Q?", "options": ["a", "b", "c", "d"], "answer_index": 2,
            "explanation": "e", "source": "s"}
    bad = {"topic": name, "question": "Q2?", "options": ["a", "a", "c", "d"], "answer_index": 0}
    qs = generate_quiz(FakeLLM({"questions": [good, bad]}), {name: [chunks[0]]}, n=5)
    assert len(qs) == 1 and qs[0]["options"][qs[0]["answer_index"]] == "c"


def test_notes_call(chunks):
    assert summarize_topic(FakeLLM(), "T", chunks) == "fake answer [1]"


def test_weak_spots_and_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    for _ in range(3):
        store.log_attempt("s", "A", "q", False)
        store.log_attempt("s", "B", "q", True)
    stats = store.topic_stats("s", ["A", "B", "C"])
    mastery = dict(zip(stats["Topic"], stats["Mastery %"]))
    assert mastery["A"] < mastery["C"] < mastery["B"]
    plan = store.make_plan(stats, date.today() + timedelta(days=10))
    assert len(plan) == 10 and plan[-1]["Focus"] == "All topics"
    assert sum(p["Focus"] == "A" for p in plan) > sum(p["Focus"] == "B" for p in plan)


def test_topics_follow_headings_without_llm():
    import io
    import pymupdf
    pdf = pymupdf.open()
    for i in range(30):
        pdf.new_page().insert_textbox(pymupdf.Rect(40, 40, 560, 800), f"topic {i // 10} " + "word " * 300)
    pdf.set_toc([[1, "Basics", 1], [1, "Models", 11], [1, "Evaluation", 21]])
    ch = build_chunks([("book.pdf", pdf.tobytes())])
    topics, _ = detect_topics(ch, None)
    assert list(topics) == ["Basics", "Models", "Evaluation"]


def _quota_error(text):
    from google.genai import errors
    return errors.ClientError(429, {"error": {"code": 429, "message": text, "status": "RESOURCE_EXHAUSTED"}})


def test_rate_limit_messages_and_backup_model():
    from studybuddy.llm import LLM, LLMError
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.append(model)
            if model == "main":
                raise _quota_error("Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
            return type("R", (), {"text": "hi"})()

    llm = LLM("key", "main", rpm=6000, use_cache=False)
    llm.client = type("C", (), {"models": FakeModels()})()
    with pytest.raises(LLMError, match="Daily"):
        llm.generate("x")
    assert calls == ["main"]  # no wasteful retries
    llm = LLM("key", "main", backup_model="backup", rpm=6000, use_cache=False)
    llm.client = type("C", (), {"models": FakeModels()})()
    assert llm.generate("x") == "hi" and calls[-2:] == ["main", "backup"]


def test_demo_mode_runs_every_feature(chunks):
    from studybuddy.mock import MockLLM
    llm = MockLLM()
    topics, _ = detect_topics(chunks, llm)
    assert topics
    text, used = answer_question(llm, Index(chunks), "What is overfitting?", [], "Clear & concise")
    assert used and "Demo mode" in text
    name = next(iter(topics))
    assert "Key points" in summarize_topic(llm, name, [chunks[i] for i in topics[name]])
    qs = generate_quiz(llm, {t: [chunks[i] for i in idx] for t, idx in topics.items()}, n=5)
    assert len(qs) == 5 and all(q["question"].startswith("Fill in the blank") for q in qs)


def test_cache_avoids_second_request(tmp_path, monkeypatch):
    from studybuddy import llm as llm_mod
    monkeypatch.setattr(llm_mod, "CACHE_DB", tmp_path / "c.db")
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.append(1)
            return type("R", (), {"text": "answer"})()

    l = llm_mod.LLM("key", "m", rpm=6000)
    l.client = type("C", (), {"models": FakeModels()})()
    assert l.generate("same") == l.generate("same") == "answer" and len(calls) == 1


# ------------------------------------------------------------------ Groq
class _Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, str(body)

    def json(self):
        return self._body


def _groq(monkeypatch, responses, model="openai/gpt-oss-120b", **kw):
    from studybuddy import llm as m
    sent, queue = [], list(responses)
    monkeypatch.setattr(m.httpx, "post", lambda url, headers, json, timeout: (sent.append((url, headers, json)), queue.pop(0))[1])
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    return m.GroqLLM("gsk_test", model, rpm=6000, use_cache=False, **kw), sent


def _ok(text):
    return _Resp(200, {"choices": [{"message": {"content": text}}]})


def _err(code, message, headers=None):
    return _Resp(code, {"error": {"message": message}}, headers)


def test_groq_request_format(monkeypatch):
    llm, sent = _groq(monkeypatch, [_ok('{"a": 1}')])
    assert llm.generate_json("give json", system="sys") == {"a": 1}
    url, headers, body = sent[0]
    assert url.endswith("/openai/v1/chat/completions") and headers["Authorization"] == "Bearer gsk_test"
    assert body["response_format"] == {"type": "json_object"} and body["reasoning_effort"] == "low"
    assert body["messages"][0] == {"role": "system", "content": "sys"}


def test_groq_non_reasoning_model_has_no_reasoning_param(monkeypatch):
    llm, sent = _groq(monkeypatch, [_ok("hi")], model="llama-3.3-70b-versatile")
    llm.generate("x")
    assert "reasoning_effort" not in sent[0][2]


def test_groq_per_minute_limit_waits_and_retries(monkeypatch):
    llm, sent = _groq(monkeypatch, [_err(429, "Rate limit reached ... tokens per minute (TPM) ... Please try again in 7.66s.",
                                         {"retry-after": "8"}), _ok("done")])
    assert llm.generate("x") == "done" and len(sent) == 2


def test_groq_daily_limit_message(monkeypatch):
    from studybuddy.llm import LLMError
    llm, sent = _groq(monkeypatch, [_err(429, "Rate limit reached on requests per day (RPD): Limit 1000. Please try again in 14m24s.")])
    with pytest.raises(LLMError, match="Daily"):
        llm.generate("x")
    assert len(sent) == 1


def test_groq_bad_key_and_helpers(monkeypatch):
    from studybuddy.llm import LLMError, parse_wait
    llm, _ = _groq(monkeypatch, [_err(401, "Invalid API Key")])
    with pytest.raises(LLMError, match="Invalid API Key"):
        llm.generate("x")
    assert parse_wait("try again in 2m59.5s") == pytest.approx(179.5) and parse_wait("nothing") is None


def test_provider_budgets_and_factory():
    from studybuddy.llm import GeminiLLM, GroqLLM, make_llm
    g = make_llm("Groq", "k")
    assert isinstance(g, GroqLLM) and g.quiz_chars < make_llm("Google Gemini", "k").quiz_chars
    assert isinstance(make_llm("Google Gemini", "k"), GeminiLLM)


def test_math_rendering():
    from studybuddy.mathfmt import render_math
    assert render_math(r"Let \(x^2\) be small") == "Let $x^2$ be small"
    assert "$$\nE = mc^2\n$$" in render_math(r"Energy: \[ E = mc^2 \]")
    assert "\\begin{aligned}a&=b\\end{aligned}" in render_math("\\begin{align}a&=b\\end{align}")
    bare = render_math("Definition:\n[ \\forall a,b,c \\in A,\\ (a,b)\\in R \\Rightarrow (a,c)\\in R ]\nDone")
    assert "$$\n\\forall" in bare and "\n[ " not in bare
    assert render_math("No math here [1] and (a, b)") == "No math here [1] and (a, b)"
    assert render_math("price $5") == "price $5"


def test_slide_pdf_titles_and_concept_topics():
    """Slide-style PDF: big titles detected, EXAMPLE/EXERCISE slides inherit the concept, related slides merge."""
    import pymupdf
    titles = ["REFLEXIVE RELATION", "EXAMPLE", "DIRECTED GRAPH OF A REFLEXIVE RELATION",
              "SYMMETRIC RELATION", "EXERCISE", "DIRECTED GRAPH OF A SYMMETRIC RELATION",
              "TRANSITIVE RELATION", "EXAMPLE"]
    pdf = pymupdf.open()
    for n, t in enumerate(titles, 1):
        pg = pdf.new_page(width=720, height=540)
        pg.insert_text((50, 60), t, fontsize=32)
        for k in range(4):
            pg.insert_text((50, 120 + 40 * k), f"Body line {k} for slide {n} with some text.", fontsize=20)
        pg.insert_text((50, 520), str(n), fontsize=12)
    ch = build_chunks([("lecture.pdf", pdf.tobytes())])
    assert [c.heading for c in ch][:2] == ["REFLEXIVE RELATION", "REFLEXIVE RELATION"]  # EXAMPLE inherited
    topics, _ = detect_topics(ch, None)
    assert list(topics) == ["Reflexive Relation", "Symmetric Relation", "Transitive Relation"]


def test_ai_topic_prompt_uses_titles_and_target(chunks):
    prompts = []

    class Spy:
        def generate_json(self, prompt, **kw):
            prompts.append(prompt)
            return {"topics": [{"name": "All", "segments": list(range(40))}]}

    detect_topics(chunks, Spy(), target=4)
    assert "about 4 topics" in prompts[0] and "Supervised Learning" in prompts[0]


def test_book_with_chapters_gets_chapter_topics_and_skips_front_matter():
    import pymupdf
    pdf = pymupdf.open()
    pages = ["Cover page text here", "Copyright notice text", "Intro to systems of equations " * 8,
             "Gaussian elimination row reduction " * 8, "Determinants and cofactor expansion " * 8,
             "Cramer rule properties " * 8, "Vectors in n-space " * 8, "Dot product and norm " * 8]
    for t in pages:
        pdf.new_page().insert_textbox(pymupdf.Rect(40, 40, 560, 800), t)
    pdf.set_toc([[1, "Cover", 1], [1, "Copyright", 2], [1, "1: Systems", 3], [2, "1.1 Intro", 3],
                 [2, "1.2 Gauss", 4], [1, "2: Determinants", 5], [2, "2.1 Cofactors", 5], [2, "2.2 Cramer", 6],
                 [1, "3: Vectors", 7], [2, "3.1 Vectors", 7], [2, "3.2 Dot", 8]])
    ch = build_chunks([("book.pdf", pdf.tobytes())])
    assert all(not c.in_topics and not c.searchable for c in ch if c.parent in ("Cover", "Copyright"))
    topics, _ = detect_topics(ch, None)
    assert list(topics) == ["1: Systems", "2: Determinants", "3: Vectors"]
    finer, _ = detect_topics(ch, None, target=6)   # asking for more topics uses the sections instead
    assert len(finer) == 6


def test_non_study_sections_are_not_topics():
    import pymupdf
    names = ["Cover Page", "Copyright Page", "The Companion Website", "To the Student",
             "1 The Foundations: Logic and Proofs", "2 Basic Structures: Sets and Functions", "Appendices",
             "Answers to Odd-Numbered Exercises", "Suggested Readings", "Bibliography", "Index", "Photo Credits"]
    pdf = pymupdf.open()
    for n in names:
        pdf.new_page().insert_textbox(pymupdf.Rect(40, 40, 560, 800), f"{n} body text " * 12)
    pdf.set_toc([[1, n, i] for i, n in enumerate(names, 1)])
    ch = build_chunks([("discrete.pdf", pdf.tobytes())])
    parents = {c.parent for c in ch if c.searchable}
    assert parents == {"1 The Foundations: Logic and Proofs", "2 Basic Structures: Sets and Functions",
                       "Appendices", "Answers to Odd-Numbered Exercises"}      # cover..credits not searchable
    topics, _ = detect_topics(ch, None)
    assert list(topics) == ["1 The Foundations: Logic and Proofs", "2 Basic Structures: Sets and Functions",
                            "Appendices"]
    answers = [c for c in ch if c.parent.startswith("Answers")]
    assert answers and not any(c.in_topics or c.topic for c in answers)         # still searchable, never a topic
    assert Index(ch).search("Answers to Odd-Numbered Exercises body text", k=1)[0].parent.startswith("Answers")


# ---------------------------------------------------------------- AI-driven topic selection
def _book(titles):
    import pymupdf
    pdf = pymupdf.open()
    for t in titles:
        pdf.new_page().insert_textbox(pymupdf.Rect(40, 40, 560, 800), f"{t} body text " * 15)
    pdf.set_toc([[1, t, i] for i, t in enumerate(titles, 1)])
    return build_chunks([("book.pdf", pdf.tobytes())])


class _OutlineLLM:
    """Fake AI: reads the outline in the prompt and classifies units by simple rules."""
    def __init__(self, exclude_words=(), reference_words=(), explicit_topics=True):
        self.ex, self.ref, self.explicit, self.prompt = exclude_words, reference_words, explicit_topics, ""

    def generate_json(self, prompt, **kw):
        import re
        self.prompt = prompt
        topics, exclude, reference = [], [], []
        for i, label in re.findall(r"^(\d+): (.*)$", prompt.split("OUTLINE:")[1], re.M):
            if any(w in label for w in self.ex):
                exclude.append(int(i))
            elif any(w in label for w in self.ref):
                reference.append(int(i))
            elif self.explicit:
                topics.append({"name": label.split(" (")[0].split(" - ")[0].lstrip("0123456789 "), "segments": [int(i)]})
        return {"topics": topics, "exclude": exclude, "reference": reference}


def test_ai_decides_what_is_study_content():
    ch = _book(["Welcome Aboard", "1 Logic", "2 Sets", "3 Counting", "Closing Thoughts and Thanks", "Solutions Manual"])
    llm = _OutlineLLM(exclude_words=("Welcome", "Closing"), reference_words=("Solutions",))
    topics, notes = detect_topics(ch, llm)
    assert not notes and list(topics) == ["Logic", "Sets", "Counting"]        # titles no keyword list knows about
    assert "Welcome Aboard" in llm.prompt and "Closing Thoughts" in llm.prompt  # the AI saw the front/back pages
    welcome = [c for c in ch if c.parent == "Welcome Aboard"]
    assert welcome and not any(c.in_topics or c.searchable for c in welcome)
    manual = [c for c in ch if c.parent == "Solutions Manual"]
    assert manual and all(c.searchable and not c.in_topics for c in manual)
    assert all(c.parent != "Welcome Aboard" for c in Index(ch).chunks)


def test_ai_can_rescue_a_page_the_keyword_list_dropped():
    ch = _book(["Preface to Algorithms", "1 Sorting", "2 Searching", "3 Graphs"])   # keyword list would drop 'Preface'
    topics, _ = detect_topics(ch, _OutlineLLM())  # AI puts every unit in a topic
    assert "Preface To Algorithms" in topics or "Preface to Algorithms" in topics


def test_keyword_prior_applies_when_ai_forgets_a_unit():
    ch = _book(["Cover Page", "1 Logic", "2 Sets", "3 Counting"])
    llm = _OutlineLLM(explicit_topics=True)
    llm.generate_json_orig = llm.generate_json
    llm.generate_json = lambda prompt, **kw: {"topics": [t for t in llm.generate_json_orig(prompt)["topics"]
                                                           if "Cover" not in t["name"]]}   # AI never mentions the cover
    topics, _ = detect_topics(ch, llm)
    assert "Cover Page" not in topics and not any(c.in_topics for c in ch if c.parent == "Cover Page")


def test_implausible_ai_answer_falls_back():
    ch = _book(["1 Logic", "2 Sets", "3 Counting", "4 Graphs"])
    llm = _OutlineLLM(exclude_words=("Logic", "Sets", "Counting"))             # AI excludes almost everything
    topics, notes = detect_topics(ch, llm)
    assert notes and "excluded most" in notes[0] and len(topics) == 4          # structure fallback, nothing lost


def test_ai_book_prompt_and_finer_split():
    ch = _book(["1 Logic", "2 Sets", "3 Counting"])
    llm = _OutlineLLM()
    detect_topics(ch, llm)
    assert "every chapter or appendix becomes its own topic" in llm.prompt
