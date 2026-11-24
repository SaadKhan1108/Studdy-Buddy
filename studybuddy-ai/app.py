"""StudyBuddy AI - Streamlit app.  Run with:  streamlit run app.py"""
import os
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from studybuddy import store
from studybuddy.features import MODES, answer_question, generate_quiz, summarize_topic
from studybuddy.ingest import SUPPORTED, build_chunks
from studybuddy.mathfmt import render_math
from studybuddy.mock import MockLLM
from studybuddy.llm import PROVIDERS, LLMError, make_llm, requests_made
from studybuddy.retrieval import Index
from studybuddy.topics import apply_topics, detect_topics

st.set_page_config(page_title="StudyBuddy AI", page_icon="🎓", layout="wide")

S = st.session_state
for key, default in {"chunks": None, "index": None, "topics": {}, "chat": [], "notes": {},
                     "quiz": None, "quiz_id": 0, "result": None}.items():
    S.setdefault(key, default)


def secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


# Fixed placeholders in the main area. They exist on every run (empty most of the time), so the page layout
# never shifts when a progress box or banner appears/disappears - otherwise Streamlit resets the tabs to "Ask".
status_slot = st.empty()
banner_slot = st.empty()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎓 StudyBuddy AI")
    provider = st.selectbox("AI provider", list(PROVIDERS),
                            help="Both have free tiers with different limits (see the README).")
    cfg = PROVIDERS[provider]
    typed_key = st.text_input(f"{provider} API key", type="password", key=f"key_{provider}",
                              help=f"Free key from {cfg['url']} - kept only in this session.")
    api_key = typed_key or os.environ.get(cfg["env"]) or secret(cfg["env"]) or ""
    model = st.text_input("Model", key=f"model_{provider}",
                          value=os.environ.get(cfg["env"].replace("API_KEY", "MODEL"), cfg["model"]),
                          help="Any model available on your free tier.")
    backup = st.text_input("Backup model (optional)", value="", key=f"backup_{provider}",
                           help="Used automatically if the main model hits its free-tier limit.")
    rpm = st.number_input("Requests per minute allowed", 1, 600, cfg["rpm"], key=f"rpm_{provider}",
                          help="Your free-tier limit for this model. The app spaces requests to stay under it.")
    demo = st.checkbox("Offline demo mode (no AI, no quota)",
                       help="Tests the whole app without any API requests. Answers/notes/quizzes are extracted "
                            "from your text instead of written by AI.")
    ai_topics = st.checkbox("Let AI organise topics when processing", value=True,
                            help="The AI decides what is study content (skipping covers, indexes, bibliographies...) "
                                 "and names the topics. Uses 1 request per file.")
    cache_on = st.checkbox("Reuse saved AI answers", value=True,
                           help="Repeating the exact same request costs no quota. Turn off to get a fresh quiz.")
    st.caption(f"📨 Requests sent since the app started: {requests_made()} (the free tier also has a daily cap)")
    st.caption(f"🔑 Get a free key at [{cfg['host']}]({cfg['url']})")
    if st.button("Test connection"):
        try:
            if demo:
                st.success("Demo mode is on - no connection needed.")
                raise LLMError("")
            make_llm(provider, api_key, model, backup, rpm, cache_on).generate(
                "Reply with the single word: ready", temperature=0.0, tries=1)
            st.success("Connected - key and model work.")
        except LLMError as e:
            if str(e):
                st.error(str(e))
    st.divider()
    profile = st.text_input("Student name", value="student")
    subject = st.text_input("Subject / course", value="My course",
                            help="Progress is tracked separately for each subject.")
    scope = f"{profile.strip() or 'student'}::{subject.strip() or 'course'}"
    st.divider()
    files = st.file_uploader("Upload slides, PDFs, books, notes", accept_multiple_files=True,
                             type=[e.lstrip(".") for e in SUPPORTED])
    c1, c2 = st.columns(2)
    process_btn = c1.button("Process", type="primary", disabled=not files, width="stretch")
    sample_btn = c2.button("Try sample", width="stretch")


def get_llm(required=True):
    if demo:
        return MockLLM()
    try:
        return make_llm(provider, api_key, model, backup, rpm, cache_on)
    except LLMError as e:
        if required:
            st.error(str(e))
            st.stop()
        return None


def load_materials(file_list):
    llm = get_llm(required=False) if ai_topics else None
    with status_slot.container(), st.status("Processing your material...", expanded=True) as status:
        st.write(f"📖 Reading {len(file_list)} file(s)...")
        try:
            chunks = build_chunks(file_list)
        except ValueError as e:
            status.update(label="Could not read a file", state="error")
            st.sidebar.error(str(e))
            return
        st.write(f"✂️ Split into {len(chunks)} passages.")
        st.write("🧭 Organising topics " + ("with AI (usually 10-30 seconds per file)..." if llm
                                            else "from headings and keywords (no AI)..."))
        topics, notes = detect_topics(chunks, llm)
        index = Index(chunks)  # built after topics: skipped pages (covers, indexes...) are not searchable
        status.update(label=f"Done - {len(topics)} topics found", state="complete", expanded=False)
    S.update(chunks=chunks, index=index, topics=topics, chat=[], notes={}, quiz=None, result=None)
    for n in notes:
        st.sidebar.warning(n)
    if ai_topics and llm is None:
        st.sidebar.info("No API key yet - topics come from headings. Add a key, then use ✨ in the Topics tab.")


if process_btn and files:
    load_materials([(f.name, f.getvalue()) for f in files])
if sample_btn:
    p = Path(__file__).parent / "samples" / "intro_to_machine_learning.md"
    load_materials([(p.name, p.read_bytes())])

if S.chunks:
    with st.sidebar:
        st.success(f"{len(S.chunks)} chunks · {len(S.topics)} topics")


def topic_chunks(names):
    return {t: [S.chunks[i] for i in S.topics[t]] for t in names}


def main_topic(used):
    counts = Counter(c.topic for c in used if c.topic)
    return counts.most_common(1)[0][0] if counts else ""


# ------------------------------------------------------------------- main
st.header("Your AI study assistant")
if demo:
    banner_slot.warning("🧪 **Offline demo mode** - no AI is used. Answers, notes and quizzes are extracted from "
                        "your text so you can test the app without any quota. Untick it in the sidebar for real AI.")
if not S.chunks:
    st.info("👈 Upload your slides, PDFs, documents or books and press **Process** "
            "(or click **Try sample**). Then ask questions, get notes, take quizzes and track weak spots.")
    st.stop()

tab_ask, tab_notes, tab_quiz, tab_prog, tab_topics = st.tabs(
    ["💬 Ask", "📝 Notes", "🧪 Quiz", "📊 Weak spots & exam prep", "🗂 Topics"])

# ------------------------------------------------------------------- Ask
with tab_ask:
    col_a, col_b = st.columns(2)
    mode = col_a.selectbox("Explanation style", list(MODES))
    scope_topic = col_b.selectbox("Limit to topic", ["All topics"] + list(S.topics))
    allowed = None if scope_topic == "All topics" else set(S.topics[scope_topic])

    for m in S.chat:
        with st.chat_message(m["role"]):
            st.markdown(render_math(m["content"]))
            if m.get("sources"):
                with st.expander("Sources"):
                    for n, src in enumerate(m["sources"], 1):
                        st.write(f"[{n}] {src}")

    question = st.chat_input("Ask anything about your material...")
    confused = st.button("🤔 I'm still confused - explain that more simply", disabled=not S.chat)
    if confused:
        question, mode = "Explain that more simply.", "Explain it simply"
    if question:
        llm = get_llm()
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    text, used = answer_question(llm, S.index, question, S.chat, mode, allowed)
            except LLMError as e:
                st.error(str(e))
                st.stop()
            st.markdown(render_math(text))
        sources = [f"{c.doc} - {c.loc}" + (f" ({c.heading})" if c.heading else "") for c in used]
        if confused:  # remember what was hard, for the weak-spot tracker
            last_q = next((m["content"] for m in reversed(S.chat) if m["role"] == "user"), question)
            last_sources = S.chat[-1].get("topic", "") if S.chat else ""
            store.log_confusion(scope, last_sources or main_topic(used), last_q)
        S.chat += [{"role": "user", "content": question},
                   {"role": "assistant", "content": text, "sources": sources, "topic": main_topic(used)}]
        if confused:
            st.rerun()

# ----------------------------------------------------------------- Notes
with tab_notes:
    choice = st.selectbox("Topic", ["All topics"] + list(S.topics), key="notes_choice")
    if choice == "All topics":
        st.caption(f"⚠️ 'All topics' uses at least {len(S.topics)} requests (one per topic). With a small daily "
                   "limit, generate one topic at a time. Finished topics are kept.")
    if st.button("Generate summarized notes"):
        llm = get_llm()
        targets = list(S.topics) if choice == "All topics" else [choice]
        bar = st.progress(0.0)
        try:
            for i, t in enumerate(targets):
                if t not in S.notes:
                    S.notes[t] = summarize_topic(llm, t, [S.chunks[j] for j in S.topics[t]])
                bar.progress((i + 1) / len(targets))
        except LLMError as e:
            st.error(str(e))
        bar.empty()
    shown = [t for t in (list(S.topics) if choice == "All topics" else [choice]) if t in S.notes]
    if shown:
        md = "\n\n---\n\n".join(f"# {t}\n\n{render_math(S.notes[t])}" for t in shown)
        st.download_button("⬇️ Download notes (.md)", md, file_name="study_notes.md")
        for t in shown:
            with st.expander(t, expanded=len(shown) == 1):
                st.markdown(render_math(S.notes[t]))

# ------------------------------------------------------------------ Quiz
def build_quiz(names, n, difficulty):
    llm = get_llm()
    try:
        with st.spinner("Writing questions from your material..."):
            qs = generate_quiz(llm, topic_chunks(names), n, difficulty)
    except LLMError as e:
        st.error(str(e))
        return False
    if not qs:
        st.error("The AI didn't return usable questions. Please try again.")
        return False
    S.quiz, S.result = qs, None
    S.quiz_id += 1
    return True


with tab_quiz:
    c1, c2, c3 = st.columns([2, 1, 1])
    qscope = c1.radio("Quiz on", ["Whole subject", "Selected topics"], horizontal=True)
    n_q = c2.slider("Questions", 3, 15, 5)
    diff = c3.selectbox("Difficulty", ["Mixed", "Easy", "Medium", "Hard"])
    picked = list(S.topics)
    if qscope == "Selected topics":
        picked = st.multiselect("Topics", list(S.topics), default=list(S.topics)[:1])
    if st.button("Generate quiz", type="primary", disabled=not picked):
        build_quiz(picked, n_q, diff)

    quiz, result = S.quiz, S.result
    if quiz:
        with st.form(f"quiz_form_{S.quiz_id}"):
            for i, q in enumerate(quiz):
                st.markdown(f"**Q{i + 1}.** {render_math(q['question'])}  \n<small>Topic: {q['topic']}</small>",
                            unsafe_allow_html=True)
                st.radio("Answer", q["options"], index=None, format_func=render_math, key=f"ans_{S.quiz_id}_{i}",
                         label_visibility="collapsed", disabled=result is not None)
            submitted = st.form_submit_button("Submit answers", disabled=result is not None)
        if submitted:
            result = []
            for i, q in enumerate(quiz):
                choice_ = S.get(f"ans_{S.quiz_id}_{i}")
                if choice_ is None:
                    result.append(None)  # skipped questions are not counted
                    continue
                ok = q["options"].index(choice_) == q["answer_index"]
                store.log_attempt(scope, q["topic"], q["question"], ok)
                result.append(ok)
            S.result = result
            st.rerun()
        if result is not None:
            answered = [r for r in result if r is not None]
            st.subheader(f"Score: {sum(answered)} / {len(answered)} answered")
            for i, (q, r) in enumerate(zip(quiz, result), 1):
                icon = "⏭️ Skipped" if r is None else ("✅ Correct" if r else "❌ Wrong")
                with st.expander(f"Q{i}: {icon}"):
                    st.markdown("**Answer:** " + render_math(q["options"][q["answer_index"]]))
                    st.markdown(render_math(q["explanation"]))
                    if q["source"]:
                        st.caption(f"Source: {q['source']}")
            st.info("Your results were saved - see the **Weak spots** tab.")

# --------------------------------------------------------- Weak spots tab
with tab_prog:
    stats = store.topic_stats(scope, list(S.topics))
    st.subheader("How well do you know each topic?")
    st.dataframe(stats.drop(columns="weakness"), hide_index=True, width="stretch")
    if stats["Attempts"].sum() or stats["Confusions"].sum():
        st.bar_chart(stats.set_index("Topic")["Mastery %"])
    else:
        st.caption("Take a quiz or ask questions and your weak spots will appear here.")

    weakest = stats.sort_values("weakness", ascending=False)["Topic"].head(3).tolist()
    st.markdown("**Focus on:** " + ", ".join(weakest))
    if st.button("🎯 Generate a practice quiz on my weakest topics"):
        if build_quiz(weakest, 8, "Mixed"):
            st.success("Quiz ready - open the **Quiz** tab.")

    st.divider()
    st.subheader("Exam prep plan")
    exam = st.date_input("Exam date", value=date.today() + timedelta(days=14), min_value=date.today())
    plan = store.make_plan(stats, exam)
    if plan:
        st.dataframe(pd.DataFrame(plan), hide_index=True, width="stretch")
        st.caption("Weaker topics get more days and are revisited in rotation. "
                   "The plan updates as your quiz results change.")
    else:
        st.write("Pick an exam date in the future to get a plan.")
    with st.expander("Reset my progress for this subject"):
        if st.button("Delete progress"):
            store.reset(scope)
            st.rerun()

# ------------------------------------------------------------ Topics tab
with tab_topics:
    st.write("Topics were detected automatically from your material's structure. "
             "**Rename** a topic, **delete** a row, or give two topics the same name to **merge** them.")
    df = pd.DataFrame({"Topic": list(S.topics), "Chunks": [len(v) for v in S.topics.values()]})
    edited = st.data_editor(df, disabled=["Chunks"], num_rows="dynamic", hide_index=True,
                            width="stretch", key="topic_editor")
    target = st.number_input("Aim for about this many topics (0 = automatic)", 0, 30, 0)
    if st.button("✨ Improve topic names with AI", help="Asks Gemini to regroup and name the topics (takes a few seconds)."):
        llm = get_llm()
        with st.spinner("Asking Gemini to organise your topics..."):
            new_topics, notes = detect_topics(S.chunks, llm, target or None)
        for n in notes:
            st.warning(n)
        if not notes:
            S.topics, S.notes, S.index = new_topics, {}, Index(S.chunks)
            st.rerun()
    if st.button("Save topic changes"):
        names = list(S.topics)
        new = {}
        for idx, row in edited.iterrows():
            name = str(row["Topic"]).strip()
            if idx in range(len(names)) and name and name != "nan":
                existing = next((k for k in new if k.lower() == name.lower()), name)
                new.setdefault(existing, []).extend(S.topics[names[idx]])
        if new:
            S.topics = {k: sorted(v) for k, v in new.items()}
            apply_topics(S.chunks, S.topics)
            S.notes = {}
            st.rerun()
        else:
            st.error("Keep at least one topic.")
