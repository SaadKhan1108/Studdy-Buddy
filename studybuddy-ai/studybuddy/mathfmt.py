"""Make LaTeX math from the AI render properly in Streamlit (KaTeX understands $...$ and $$...$$).

Models often write \\( ... \\) and \\[ ... \\]; Streamlit's Markdown strips those backslashes, which
is why formulas showed up as raw text. We convert every common form to $ / $$ before display.
"""
from __future__ import annotations

import re

_ENV = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}(.+?)\\end\{\1\}", re.S)
_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.S)
_INLINE = re.compile(r"\\\((.+?)\\\)", re.S)
# a line that is only "[ ...latex... ]" (backslashes around the bracket were lost)
_BARE = re.compile(r"^[ \t]*\[[ \t]+(.*\\[A-Za-z]+.*?)[ \t]+\][ \t]*$", re.M)


def _env(m):
    kind, body = m.group(1).rstrip("*"), m.group(2).strip()
    if kind in ("align", "eqnarray"):
        body = "\\begin{aligned}" + body + "\\end{aligned}"  # KaTeX supports 'aligned', not 'align'
    return f"\n$$\n{body}\n$$\n"


def render_math(text: str) -> str:
    if not text or "\\" not in text:
        return text
    text = _ENV.sub(_env, text)
    text = _DISPLAY.sub(lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", text)
    text = _INLINE.sub(lambda m: f"${m.group(1).strip()}$", text)
    text = _BARE.sub(lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", text)
    return text


MATH_RULE = ("Write all mathematics in LaTeX: use $...$ for inline formulas and $$...$$ on their own lines "
             "for display equations. Never use \\( \\) or \\[ \\] delimiters. Keep all other text plain Markdown.")
