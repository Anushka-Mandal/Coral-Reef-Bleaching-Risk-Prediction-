"""LLM explanations grounded in the forecast, live numbers and reef reports.

How data is fed to the model (one request per briefing):
  1. system prompt - role and strict grounding rules
  2. documents     - [Data]: the numbers as plain sentences (NOAA observations, the model's
                     forecast and its measured skill), built by grounding.facts_to_text
                   - [S1]..[S6]: the most relevant report passages from rag.py
                   All documents have the Citations API enabled, so every cited claim comes
                   back with a pointer to real text in these documents.
  3. the task      - a briefing, or the user's question
Claude streams its answer; citation markers ([Data], [S2], ...) are added after each cited
passage, and grounding.verify() then checks every number and flags uncited factual claims.

If no API key is configured, or the API call fails, `offline_briefing` writes a
plain rule-based briefing from the same facts, so the demo never breaks.
"""

import json
import logging
import os

from . import grounding
from .rag import pretty_title

log = logging.getLogger("coralwatch.llm")

MODEL = os.environ.get("CORALWATCH_MODEL", "claude-opus-5")
# Free option: an open-source model running locally in Ollama (no key, no cost, works offline)
LOCAL_MODEL = os.environ.get("CORALWATCH_LOCAL_MODEL", "qwen2.5:3b")
OLLAMA_URL = os.environ.get("CORALWATCH_OLLAMA_URL", "http://127.0.0.1:11434")
STRICT = os.environ.get("CORALWATCH_STRICT", "1") != "0"   # remove unverifiable sentences

LOCAL_SYSTEM_PROMPT = """You are CoralWatch. You explain coral bleaching risk on the Great Barrier Reef using ONLY the documents given.

Strict rules:
1. Use only facts written in the documents. Do not add anything from your own knowledge.
2. Copy every number exactly as written in the documents. Never calculate, estimate or change numbers.
3. End EVERY sentence that states a fact with its document tag, exactly like [Data] or [S2]. Only use tags that exist.
4. If the documents do not answer something, write: "The data and reports provided don't cover this."
5. Satellites measure heat stress, not bleaching. Only say corals bleached or died if an [S] document says so, and give its year.
6. Summarise in your own words. Never copy whole passages, document titles or the word DOCUMENT.
7. At most 3 short sentences under each heading.

For a briefing, use exactly these four headings, about 150 words in total:
### What the satellites show
### Next 7 days
### What the reports tell us
### Suggested actions"""

SYSTEM_PROMPT = """You are CoralWatch, an assistant that explains coral bleaching risk on the northern and central Great Barrier Reef to reef managers and students.

You receive documents: a "CoralWatch data" document with NOAA satellite values, the model forecast and its measured skill, and report passages from AIMS, the Marine Monitoring Program, the Reef Authority and NOAA.

Grounding rules - these matter more than anything else:
- Say only what the documents support, and cite them. Every sentence with a number, a date, a trend, a bleaching or mortality statement, or a forecast must be backed by a citation to a document.
- Copy numbers exactly as they appear in the documents. Never calculate, estimate, round differently or invent figures, and don't use numbers from general knowledge.
- If the documents don't answer something, say "The data and reports provided don't cover this" instead of guessing.
- Satellites measure heat stress, not bleaching. Only say corals bleached or died if a report passage says so, and name the year it refers to.
- Report passages may describe earlier years; say which year when you use one.
- State the forecast's uncertainty using its measured skill from the data document, and mention if the date was in the model's training data.
- Plain language, short sentences. Explain DHW (accumulated heat stress) the first time you use it.

Format for a briefing (these four markdown headings, about 220 words in total):
### What the satellites show
### Next 7 days
### What the reports tell us
### Suggested actions
Suggested actions may be general good practice, but tie them to the documented situation.

For a question, answer directly in under 200 words with the same rules."""


def build_content(facts, sources, question=None):
    """User-message content: the data document, the report passages (all citable) and the task."""
    docs = [{
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": grounding.facts_to_text(facts)},
        "title": f"CoralWatch data: {facts['zone']['name']}, {facts['date']}",
        "context": "Computed from NOAA Coral Reef Watch satellite data and the CoralWatch forecast model.",
        "citations": {"enabled": True},
    }]
    for s in sources:
        where = f", p. {s['page']}" if s.get("page") else ""
        year = f" ({s['published']})" if s.get("published") else ""
        docs.append({
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": s["text"]},
            "title": f"{s['source']} - {pretty_title(s['title'])}{year}{where}",
            "citations": {"enabled": True},
        })
    task = (f"Question about the {facts['zone']['name']} on {facts['date']}: {question}" if question
            else f"Write the briefing for the {facts['zone']['name']} as of {facts['date']}.")
    return docs + [{"type": "text", "text": task}]


def marker_for(document_index):
    return "[Data]" if document_index == 0 else f"[S{document_index}]"


def configured():
    """True if a real LLM (Claude or the free local model) is available."""
    return provider()[0] != "offline"


def _ollama_up():
    try:
        import httpx
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        return r.status_code == 200 and any(m.get("name", "").startswith(LOCAL_MODEL) for m in r.json().get("models", []))
    except Exception:
        return False


def provider():
    """("claude", model) if an Anthropic key is set, else ("local", model) if Ollama is running
    with the model, else ("offline", None). CORALWATCH_LLM=claude|local|offline forces a choice."""
    choice = os.environ.get("CORALWATCH_LLM", "auto")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if choice == "claude" or (choice == "auto" and has_key):
        return ("claude", MODEL) if has_key else ("offline", None)
    if choice in ("local", "auto") and _ollama_up():
        return "local", LOCAL_MODEL
    return "offline", None


LOCAL_MAX_SOURCES = 4        # small local models: fewer, shorter passages = faster and less to confuse
LOCAL_MAX_WORDS = 140


def local_sources(sources):
    return [{**s, "text": " ".join(s["text"].split()[:LOCAL_MAX_WORDS])} for s in sources[:LOCAL_MAX_SOURCES]]


def build_local_messages(facts, sources, question=None):
    """Prompt for the local model: documents tagged [Data] / [S1].. inline (no Citations API)."""
    parts = [f'<document tag="[Data]" about="CoralWatch data, {facts["zone"]["name"]}, {facts["date"]}">\n'
             f'{grounding.facts_to_text(facts)}\n</document>']
    for k, s in enumerate(local_sources(sources), 1):
        year = f", {s['published']}" if s.get("published") else ""
        parts.append(f'<document tag="[S{k}]" about="{s["source"]}{year}">\n{s["text"]}\n</document>')
    task = (f"QUESTION about the {facts['zone']['name']} on {facts['date']}: {question}" if question
            else f"TASK: Write the briefing for the {facts['zone']['name']} as of {facts['date']}. Tag every factual sentence.")
    return [{"role": "system", "content": LOCAL_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts) + "\n\n" + task}]


def _stream_local(facts, sources, question):
    """Free local model via Ollama, with strict-mode clean-up of unverifiable sentences."""
    import httpx
    answer = []
    body = {"model": LOCAL_MODEL, "messages": build_local_messages(facts, sources, question), "stream": True,
            "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 420},  # low randomness, capped length
            "keep_alive": "30m"}
    with httpx.stream("POST", f"{OLLAMA_URL}/api/chat", json=body, timeout=300.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            piece = msg.get("message", {}).get("content", "")
            if piece:
                answer.append(piece)
                yield "delta", piece
            if msg.get("done"):
                break
    text = grounding.tidy_local(grounding.normalize_tags("".join(answer)))
    srcs = local_sources(sources)
    facts_text, src_texts = grounding.facts_to_text(facts), [s["text"] for s in srcs]
    # small models cite unreliably: attribute each sentence to the document that supports it
    text, auto = grounding.attribute(text, [("Data", facts_text)] + [(f"S{k}", t) for k, t in enumerate(src_texts, 1)])
    report = grounding.verify(text, facts_text, src_texts)
    removed = 0
    if STRICT and not report["grounded"]:
        text, removed = grounding.remove_unsupported(text, report)
        report = grounding.verify(text, facts_text, src_texts)
    yield "replace", {"text": text, "removed": removed}
    report["removed_sentences"] = removed
    report["auto_attributed"] = auto
    yield "verify", report
    yield "done", {"mode": "llm", "model": f"{LOCAL_MODEL} (free, local)", "provider": "local"}


def stream_explanation(facts, sources, question=None):
    """Yields ("delta", text) chunks, then ("verify", report) and ("done", info).
    Falls back to the offline briefing on any failure."""
    kind, _ = provider()
    if kind == "offline":
        yield from _offline(facts, sources, question,
                            reason="No LLM available (no Anthropic key, and Ollama isn't running) - showing the offline briefing.")
        return
    if kind == "local":
        try:
            yield from _stream_local(facts, sources, question)
        except Exception as e:
            log.warning("Local model failed: %s", e)
            yield from _offline(facts, sources, question, reason="The local model failed - showing the offline briefing.")
        return

    import anthropic

    client = anthropic.Anthropic(max_retries=2, timeout=120.0)
    answer, sent = [], False
    try:
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": build_content(facts, sources, question)}],
        ) as stream:
            block_cites, block_text, blocks = [], [], []
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block_cites, block_text = [], []
                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        sent = True
                        answer.append(delta.text)
                        block_text.append(delta.text)
                        yield "delta", delta.text
                    elif delta.type == "citations_delta":
                        m = marker_for(delta.citation.document_index)
                        if m not in block_cites:
                            block_cites.append(m)
                elif etype == "content_block_stop":
                    if block_text:
                        blocks.append(("".join(block_text), list(block_cites)))
                    if block_cites:
                        marks = "".join(block_cites)   # cite right after the passage it supports
                        answer.append(marks)
                        yield "delta", marks
                    block_cites, block_text = [], []
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            if not sent:
                yield from _offline(facts, sources, question, reason="The model declined this request - showing the offline briefing.")
                return
            yield "delta", "\n\n*(The response was stopped early.)*"
        yield "verify", grounding.verify("".join(answer), grounding.facts_to_text(facts),
                                         [s["text"] for s in sources], blocks=blocks)
        yield "done", {"mode": "llm", "model": final.model, "input_tokens": final.usage.input_tokens,
                       "output_tokens": final.usage.output_tokens}
    except anthropic.AuthenticationError:
        yield from _offline(facts, sources, question, reason="The API key was rejected - showing the offline briefing.")
    except anthropic.RateLimitError:
        yield from _offline(facts, sources, question, reason="The LLM is rate-limited right now - showing the offline briefing.")
    except anthropic.APIStatusError as e:
        log.warning("Claude API error %s: %s", e.status_code, e.message)
        yield from _offline(facts, sources, question, reason=f"LLM error ({e.status_code}) - showing the offline briefing.")
    except anthropic.APIConnectionError:
        yield from _offline(facts, sources, question, reason="Couldn't reach the LLM - showing the offline briefing.")


def _offline(facts, sources, question, reason):
    text = offline_briefing(facts, sources)
    if question:
        text = f"*Questions need the LLM. Here is the standard briefing instead.*\n\n{text}"
    for para in text.split("\n"):
        yield "delta", para + "\n"
    yield "verify", grounding.verify(text, grounding.facts_to_text(facts), [x["text"] for x in sources])
    yield "done", {"mode": "offline", "reason": reason}


def offline_briefing(facts, sources):
    zone = facts["zone"]["name"]
    o = facts.get("observed") or {}
    lines = ["### What the satellites show"]
    if o:
        lines.append(
            f"On {facts['date']}, {o.get('reefs_at_alert', 0)} of {o.get('reefs_total', 0)} reefs in the {zone} "
            f"were at Bleaching Alert Level 1 or higher. The worst level was {o.get('worst_level', 'unknown')}, "
            f"and Degree Heating Weeks (DHW, accumulated heat stress) reached {o.get('dhw_max', 0)} °C-weeks. [Data]")
        if "dhw_change_7d" in o:
            trend = "rising" if o["dhw_change_7d"] > 0.1 else "falling" if o["dhw_change_7d"] < -0.1 else "steady"
            lines.append(f"Over the past week DHW has been {trend} ({o['dhw_change_7d']:+} on average). [Data]")
    f = facts.get("forecast")
    lines.append("\n### Next 7 days")
    if f:
        fz = f["zone"]
        lines.append(f"The {f['model']} model expects {fz.get('reefs_at_alert', 0)} reefs at Alert Level 1+ "
                     f"on {f['target_date']} (worst level: {fz.get('worst_level')}). [Data]")
        if f.get("skill"):
            s = f["skill"]
            mt = s.get("model_test_7d") or s.get("model_test")
            pt = s.get("persistence_test_7d") or s.get("persistence_test")
            lines.append(f"On unseen test seasons its 7-day accuracy was {mt['accuracy'] * 100:.1f}% "
                         f"vs {pt['accuracy'] * 100:.1f}% for assuming no change. [Data]")
    else:
        lines.append("No forecast is available for this date.")
    lines.append("\n### What the reports tell us")
    if sources:
        for k, s in enumerate(sources[:3], 1):
            snippet = " ".join(s["text"].split()[:40])
            lines.append(f"- {snippet}… [S{k}]")
    else:
        lines.append("No relevant report passages were found.")
    lines.append("\n### Suggested actions")
    level = o.get("worst_level") or ""
    if "Alert" in level:
        lines.append("- Prioritise in-water and aerial surveys of reefs at Alert Level 2.\n"
                     "- Reduce local stressors (e.g. anchoring, runoff) on the most exposed reefs.\n"
                     "- Keep checking DHW daily; NOAA guidance says 8 or more means coral death is likely. [Data]")
    elif level in ("Watch", "Warning"):
        lines.append("- Keep watching the daily heat-stress maps and plan surveys in case Alerts develop.")
    else:
        lines.append("- No action needed beyond routine monitoring.")
    return "\n".join(lines)
