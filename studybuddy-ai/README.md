# 🎓 StudyBuddy AI

An AI study assistant for university students. Upload your slides, PDFs, Word documents or books and
**ask questions, get concepts explained, generate summarized notes, take topic-based quizzes, and let the app
track your weak spots and plan your exam revision.**

> Built for the **Build Your First AI App 2026** hackathon (Yasir Insights / Devpost).

## The problem
Students collect slides, PDFs and textbook chapters across many courses. Before exams they re-read everything,
without knowing which topics they actually understand. StudyBuddy turns the student's *own* material into a tutor,
a note-taker and a quiz-master, and shows where to spend revision time.

## Features
| Feature | How it works |
|---|---|
| **Ask questions about your material** | Retrieval-augmented generation (RAG): the most relevant passages are found and given to the AI, which answers only from them and cites `[1]`, `[2]` with file + page/slide. If the answer isn't in your material it says so. |
| **Understand hard concepts** | Choose *Clear & concise / Explain it simply / Go deeper*. The **"I'm still confused"** button re-explains with an analogy and records the confusion. |
| **Summarized notes** | Per topic (or all topics): definitions, formulas, key terms, examples, exam tips, quick recap. Long topics are condensed in parts first (map-reduce). Download as Markdown. |
| **Automatic topic detection** | The app builds an outline of the whole file (chapters of a book, slide titles, headings), and the AI classifies every part as *study content*, *reference only* (answer keys, glossary) or *excluded* (cover, copyright, preface, bibliography, index...), then groups and names the topics. One request per file. Without an AI key, or if the AI fails, it falls back to the document's own bookmarks/headings plus a small keyword list. Topics can be renamed, merged or deleted. |
| **Quizzes** | Whole subject or selected topics, 3-15 questions, choice of difficulty. Multiple choice with plausible distractors, explanations and a source reference for every question. |
| **Weak-spot tracking** | Every answer is saved. Mastery per topic = recency-weighted accuracy (older answers count less); "I'm confused" questions add to a topic's weakness. |
| **Exam prep** | Enter your exam date: weaker topics get more days and are revisited in rotation; the last day is a mixed mock quiz. One click generates a practice quiz on your 3 weakest topics. |

Supported files: `.pdf`, `.pptx`, `.docx`, `.txt`, `.md` (text-based; scanned PDFs need OCR and are not supported).

## AI implementation
- **LLM:** your choice of **Google Gemini** (default `gemini-3.8-flash`, via the `google-genai` SDK) or **Groq** (default `openai/gpt-oss-120b`, via Groq's OpenAI-compatible API), both on free tiers. Pick the provider and model in the sidebar. It writes answers, notes, topic names and quiz questions. Quiz and topic output uses JSON mode and is validated in code (4 distinct options, valid answer index, shuffled options).
- **Retrieval:** TF-IDF (unigrams + bigrams) with cosine similarity in scikit-learn. No GPU or heavy model downloads, so it runs on free hosting.
- **Prompting:** the tutor is instructed to answer only from numbered excerpts and cite them, to reduce hallucination.
- **Non-AI logic:** file parsing, chunking, topic fallback clustering, mastery scoring and the study planner are plain Python.

## Choosing an AI provider (free tiers)
| | Google Gemini | Groq |
|---|---|---|
| Get a key | aistudio.google.com/apikey | console.groq.com/keys |
| Strengths | Very long context, strong on long documents | Very fast responses, many more requests per day |
| Limits to know | Small daily request cap on some free models (check `aistudio.google.com/usage`) | Small tokens-per-minute and tokens-per-day caps (check `console.groq.com/settings/limits`) |
| How the app adapts | Sends large passages per request | Sends smaller passages per request and waits out per-minute limits automatically |

Free-tier limits change and differ per account and model - always check your own limits page, then set **Requests per minute allowed** in the sidebar. A **Backup model** switches automatically if the main model is rate-limited. **Offline demo mode** tests the whole app with no AI requests, and **Reuse saved AI answers** avoids re-spending quota on identical requests (answers are stored in `data/` on the machine running the app; untick it if you prefer not to store them).

## Try it
### Online (judges)
Open the deployed app, choose a provider and paste a **free API key** into the sidebar (Gemini: https://aistudio.google.com/apikey, Groq: https://console.groq.com/keys - the key stays in your browser session and is never stored). To explore without any key, tick **Offline demo mode**, then click **Try sample** or upload your own files.

### Run locally
```bash
git clone <your-repo-url> && cd studybuddy-ai
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Optionally set `GEMINI_API_KEY` or `GROQ_API_KEY` as an environment variable instead of typing it each time.

### Run the tests (no API key needed)
```bash
pip install pytest && python -m pytest -q
```

### Deploy for free
Streamlit Community Cloud: push this repo to GitHub → *New app* → main file `app.py`. Do **not** commit API keys or private course files (`.gitignore` already excludes `data/`, `.env` and `secrets.toml`).

## Project structure
```
app.py                  Streamlit interface (5 tabs)
studybuddy/ingest.py    PDF/PPTX/DOCX/TXT readers + chunking with page/slide references
studybuddy/retrieval.py TF-IDF search
studybuddy/topics.py    Topic detection (structure + AI + clustering fallback)
studybuddy/features.py  Q&A with citations, notes, quiz generation
studybuddy/store.py     SQLite progress tracking, weak-spot scoring, study plan
studybuddy/llm.py       Gemini + Groq providers: pacing, caching, retries, friendly errors
studybuddy/mock.py      Offline demo mode (no AI, no quota)
tests/test_core.py      Offline tests using a fake LLM
samples/                Sample notes for a quick demo
```

## Limitations & ideas
- Scanned/image-only PDFs and text inside images are not read (OCR could be added).
- Keyword search can miss questions phrased very differently from the text; embeddings would improve this.
- Uploaded material lives in the session only; quiz progress is stored in a local SQLite file (on free hosting it may reset when the app restarts).
- Free-tier API limits apply; the app retries automatically and shows a message when the limit is hit.
- Quiz questions are AI-generated from your material - always double-check against your textbook.

## My contribution & AI assistance  *(edit this section before submitting)*
- **What I did:** *e.g. defined the idea and features, chose the stack, tested the app with my own course material, wrote the demo and documentation...*
- **What I learned:** *e.g. how RAG works, prompt design, JSON validation, Streamlit state...*
- **AI-assisted development:** this project was developed with the help of **Claude (Anthropic)**, which generated the initial code from my specification. I reviewed, ran and tested it.

## Credits
Streamlit, Google Gemini API / `google-genai`, PyMuPDF, python-pptx, python-docx, scikit-learn, pandas.
