"""Grounding: turning facts into a citable document, and checking answers for hallucinations.

Two jobs:
  facts_to_text()  writes the numbers (NOAA observations, model forecast, model skill) as short
                   plain sentences, so they can be passed to Claude as a citable document
  verify()         checks a finished answer:
                     * every number must appear in the facts or the report passages
                       (allowing rounding and fraction -> percent)
                     * factual sentences (numbers, bleaching or mortality claims) must carry a citation
                   and returns a report the website shows next to the answer
"""

import re

MARKER = re.compile(r"\[(?:Data|S\d+)\]")
LOOSE_TAG = re.compile(r"\[\s*(?:(data)|s\s*(\d+))\s*\]", re.I)


def normalize_tags(text):
    """Local models write tags loosely ("[ S2 ]", "[DATA]"); make them "[S2]" / "[Data]"."""
    return LOOSE_TAG.sub(lambda m: "[Data]" if m.group(1) else f"[S{m.group(2)}]", text)
NUMBER = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.])[-+]?\d+(?:\.\d+)?%?")
CLAIM_WORDS = re.compile(r"\b(bleach\w*|mortality|died|dead|death|coral cover|cyclone|surveyed|survey)\b", re.I)


def facts_to_text(facts):
    """One fact per sentence, so each citation points at exactly one fact."""
    zone, date = facts["zone"]["name"], facts["date"]
    lines = [f"This document gives NOAA Coral Reef Watch satellite values and the CoralWatch forecast for the {zone} on {date}."]
    o = facts.get("observed") or {}
    if o:
        lines.append(f"On {date}, {o.get('reefs_at_alert', 0)} of {o.get('reefs_total', 0)} reefs in the {zone} were at Bleaching Alert Level 1 or higher.")
        lines.append(f"On {date}, the worst alert level in the {zone} was {o.get('worst_level')}.")
        for lvl, share in (o.get("area_share_by_level") or {}).items():
            lines.append(f"On {date}, {share * 100:.1f}% of the {zone} area was at {lvl}.")
        for lvl, n in (o.get("reefs_by_level") or {}).items():
            lines.append(f"On {date}, {n} reefs in the {zone} were at {lvl}.")
        for key, label, unit in (("dhw", "Degree Heating Weeks (DHW)", "degC-weeks"), ("hotspot", "HotSpot", "degC"),
                                 ("ssta", "sea surface temperature anomaly", "degC")):
            if f"{key}_max" in o:
                lines.append(f"On {date}, the {label} in the {zone} averaged {o[f'{key}_mean']} {unit} with a maximum of {o[f'{key}_max']} {unit}.")
        if o.get("dhw_change_7d") is not None:
            lines.append(f"Over the 7 days to {date}, DHW in the {zone} changed by {o['dhw_change_7d']:+} degC-weeks on average.")
        if o.get("ssta_change_7d") is not None:
            lines.append(f"Over the 7 days to {date}, the temperature anomaly in the {zone} changed by {o['ssta_change_7d']:+} degC on average.")
    lines.append("NOAA guidance: DHW of 4 degC-weeks or more means significant bleaching is likely; 8 or more means severe bleaching and significant coral death are likely.")
    f = facts.get("forecast")
    if f:
        lines.append(f"The forecast was made by the {f['model']} model on {f['issued']}.")
        lines.append(f"The forecast quantity is: {f.get('what_is_forecast', 'the NOAA alert level')}.")
        if f.get("weather_used"):
            lines.append(f"The forecast used {f['weather_used']}.")
        z = f["zone"]
        lines.append(f"For {f['target_date']} (7 days ahead), the model forecasts {z.get('reefs_at_alert', 0)} reefs in the {zone} at Alert Level 1 or higher, worst level {z.get('worst_level')}.")
        if z.get("mean_alert_probability") is not None:
            lines.append(f"The average forecast chance of Alert across the {zone} for {f['target_date']} is {z['mean_alert_probability'] * 100:.0f}%.")
        import datetime as _dt
        for k, zz in (f.get("zone_by_lead_days") or {}).items():
            if zz.get("reefs_at_alert") is not None:
                day = (_dt.date.fromisoformat(f["issued"]) + _dt.timedelta(days=int(k))).isoformat()
                lines.append(f"For {day} ({k} day{'s' if int(k) > 1 else ''} ahead), the model forecasts {zz['reefs_at_alert']} reefs in the {zone} at Alert Level 1 or higher.")
        if f.get("date_was_in_training_data"):
            lines.append("This date was part of the model's training data, so the forecast for it is not an independent test.")
        s = f.get("skill") or {}
        mt, pt = s.get("model_test_7d") or s.get("model_test"), s.get("persistence_test_7d") or s.get("persistence_test")
        if mt and pt:
            lines.append(f"On unseen test seasons ({s.get('test_seasons', '2024-2025')}), the model's 7-day accuracy was {mt['accuracy'] * 100:.1f}% versus {pt['accuracy'] * 100:.1f}% for assuming no change.")
        real = s.get("with_real_weather_forecasts_7d")
        if real:
            lines.append(f"Using the real weather forecasts of the time ({real['period']}), the 7-day accuracy was {real['model_accuracy'] * 100:.1f}% versus {real['persistence_accuracy'] * 100:.1f}% for assuming no change.")
    else:
        lines.append(f"No model forecast is available for {date}.")
    lines.append("Satellites measure heat stress, not bleaching itself.")
    return "\n".join(lines)


def _to_float(tok):
    return float(tok.rstrip("%").replace(",", "").lstrip("+"))


def _reference_numbers(texts):
    refs = set()
    for t in texts:
        for tok in NUMBER.findall(t):
            try:
                v = _to_float(tok)
            except ValueError:
                continue
            refs.add(v)
            if 0 < abs(v) <= 1 and "." in tok:        # fractions can be quoted as percentages
                refs.add(v * 100)
    return refs


def _matches(x, refs, is_pct, is_int):
    """Whole numbers (counts, years) must match exactly, or be a documented decimal rounded;
    decimals and percentages may differ by rounding."""
    for r in refs:
        if is_int and not is_pct:
            if x == r or (r != int(r) and x == round(r)):
                return True
            continue
        if abs(x - r) <= max(0.051, 0.005 * abs(r) if abs(r) < 100 else 0.051):
            return True
        if is_pct and abs(x / 100 - r) <= 0.006:
            return True
    return False


def split_sentences(text):  # kept for analysis scripts
    """Sentences, keeping citation markers with the sentence they follow ("week. [Data] Next" ->
    "week.[Data]" + "Next")."""
    body = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    out = []
    for ln in body:
        ln = re.sub(r"([.!?])\s+((?:\[(?:Data|S\d+)\])+)", r"\1\2", ln.strip("-* ").strip())
        out += [s.strip() for s in re.split(r"(?<=[.!?\]])\s+(?=[A-Z(\-*])", ln) if s.strip()]
    return out


def cited_units(answer):
    """(text, is_cited) units. A citation marker covers the text before it back to the previous
    marker or line start; text after the last marker on a line is uncited."""
    units = []
    for line in answer.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        pos = 0
        for m in re.finditer(r"(?:\[(?:Data|S\d+)\])+", line):
            seg = line[pos:m.start()].strip(" -*")
            if seg:
                units.append((seg, True))
            pos = m.end()
        tail = line[pos:].strip(" -*")
        if tail:
            units.append((tail, False))
    return units


def _advice_lines(answer):
    """Lines under a 'Suggested actions' heading."""
    out, in_actions = [], False
    for line in answer.splitlines():
        if line.lstrip().startswith("#"):
            in_actions = "action" in line.lower()
        elif in_actions and line.strip():
            out.append(MARKER.sub("", line).strip(" -*"))
    return [ln for ln in out if ln]


DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _ref_sentences(texts):
    out = []
    for t in texts:
        for ln in t.splitlines():
            out += [x for x in re.split(r"(?<=[.!?])\s+", ln) if x.strip()]
    return out


def context_mismatches(answer, facts_text, source_texts):
    """Right number, wrong context: a sentence that pairs a date with a number must match one
    document sentence containing that same date and number (e.g. a 1-day forecast quoted
    as the 7-day forecast)."""
    refs = [(set(DATE.findall(r)), _reference_numbers([DATE.sub(" ", r)])) for r in _ref_sentences([facts_text, *source_texts])]
    bad = []
    for line in MARKER.sub(" ", answer).splitlines():
        if line.lstrip().startswith("#"):
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            dates = set(DATE.findall(sent))
            nums = _nontrivial_numbers(DATE.sub(" ", sent))
            if not dates or not nums:
                continue
            for x, pct, integer in nums:
                if not any(dates & rd and _matches(x, rn, pct, integer) for rd, rn in refs):
                    bad.append(sent.strip())
                    break
    return bad


def verify(answer, facts_text, source_texts, blocks=None):
    """Hallucination check of a finished answer. Returns a report for the UI and logs.
    blocks: optional [(text, markers)] from the stream (exact Citations API block boundaries);
    otherwise units are inferred from marker positions."""
    refs = _reference_numbers([facts_text, *source_texts])
    clean = MARKER.sub(" ", answer)
    checked, unverified = 0, []
    for line in clean.splitlines():
        if line.lstrip().startswith("#"):
            continue                                   # headings like "Next 7 days"
        for tok in NUMBER.findall(line):
            try:
                x = _to_float(tok)
            except ValueError:
                continue
            if "." not in tok and "%" not in tok and abs(x) <= 14:
                continue                               # small counts: levels, lead days, list items
            checked += 1
            if not _matches(x, refs, tok.endswith("%"), "." not in tok):
                unverified.append(tok)
    if blocks is not None:
        units = []
        for text, marks in blocks:
            body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#")).strip(" -*\n")
            if body:
                units.append((body, bool(marks)))
    else:
        units = cited_units(answer)
    advice = _advice_lines(answer)          # recommendations don't need citations (numbers still checked)
    units = [(t, c) for t, c in units if not any(t in ln or ln in t for ln in advice)]
    factual = [(t, c) for t, c in units if NUMBER.search(t) or CLAIM_WORDS.search(t)]
    uncited = [t for t, c in factual if not c]
    report = {
        "numbers_checked": checked,
        "numbers_verified": checked - len(unverified),
        "unverified_numbers": sorted(set(unverified)),
        "factual_sentences": len(factual),
        "cited_factual_sentences": len(factual) - len(uncited),
        "uncited_claims": [u[:160] for u in uncited[:3]],
    }
    report["context_mismatches"] = [m[:160] for m in context_mismatches(answer, facts_text, source_texts)[:3]]
    report["grounded"] = not report["unverified_numbers"] and not uncited and not report["context_mismatches"]
    return report


def remove_unsupported(answer, report):
    """Strict mode: drop sentences that contain an unverified number or an uncited factual claim.
    Returns (cleaned_text, removed_count). Headings and advice are kept."""
    bad_nums = set(report.get("unverified_numbers", []))
    bad_context = [c.strip() for c in report.get("context_mismatches", [])]
    advice = _advice_lines(answer)
    out, removed = [], 0
    for line in answer.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            continue
        bullet = re.match(r"^\s*([-*]\s+)", line)
        prefix = bullet.group(1) if bullet else ""
        body = line[len(bullet.group(0)):] if bullet else line
        body = re.sub(r"([.!?])\s+((?:\[(?:Data|S\d+)\])+)", r"\1\2", body)
        keep = []
        for sent in re.split(r"(?<=[.!?\]])\s+(?=[A-Z(])", body):
            plain = MARKER.sub(" ", sent)
            is_advice = any(a and (a in sent or sent.strip() in a) for a in advice)
            has_bad = any(re.search(rf"(?<![\w.]){re.escape(n)}(?![\w])", plain) for n in bad_nums)
            factual = NUMBER.search(plain) or CLAIM_WORDS.search(plain)
            wrong_ctx = any(c and c[:60] in plain for c in bad_context)
            if has_bad or wrong_ctx or (factual and not MARKER.search(sent) and not is_advice):
                removed += 1
                continue
            keep.append(sent)
        if keep:
            out.append(prefix + " ".join(keep))
    return "\n".join(out), removed


STOP = set("""the a an and or of in on at to for from by with was were is are be been this that these those it its
as than more most over under into about after before between during their there which who what when where while not
no has have had will would can could may might also per each all any some such""".split())


def _content_words(text):
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOP}


def _nontrivial_numbers(text):
    out = []
    for tok in NUMBER.findall(text):
        try:
            x = _to_float(tok)
        except ValueError:
            continue
        if "." in tok or "%" in tok or abs(x) > 14:
            out.append((x, tok.endswith("%"), "." not in tok))
    return out


def attribute(answer, documents, min_overlap=0.35):
    """Automatic attribution for models that don't cite reliably: give each untagged factual
    sentence the tag of a document that supports it - every number in the sentence appears in
    that document and enough of its key words do too. Unsupported sentences stay untagged
    (so verify/strict mode flag or remove them). documents: [(tag, text)].
    Returns (text, number_of_sentences_tagged)."""
    docs = [(tag, _content_words(t), _reference_numbers([t])) for tag, t in documents]
    out, tagged = [], 0
    for line in answer.splitlines():
        if line.lstrip().startswith("#"):
            out.append(MARKER.sub("", line).rstrip())       # no tags on headings
            continue
        if not line.strip():
            out.append(line)
            continue
        bullet = re.match(r"^\s*([-*]\s+)", line)
        prefix = bullet.group(1) if bullet else ""
        body = line[len(bullet.group(0)):] if bullet else line
        body = re.sub(r"([.!?])\s+((?:\[(?:Data|S\d+)\])+)", r"\1\2", body)
        sents = []
        for sent in re.split(r"(?<=[.!?\]])\s+(?=[A-Z(])", body):
            plain = MARKER.sub(" ", sent)
            if MARKER.search(sent) or not (NUMBER.search(plain) or CLAIM_WORDS.search(plain)):
                sents.append(sent)
                continue
            words, nums = _content_words(plain), _nontrivial_numbers(plain)
            best, best_score = None, 0.0
            for tag, dwords, drefs in docs:
                if not all(_matches(x, drefs, pct, integer) for x, pct, integer in nums):
                    continue
                score = len(words & dwords) / max(1, len(words))
                if score > best_score:
                    best, best_score = tag, score
            if best and best_score >= min_overlap:
                sent = sent.rstrip() + f" [{best}]"
                tagged += 1
            sents.append(sent)
        out.append(prefix + " ".join(sents))
    return "\n".join(out), tagged


HEADINGS = ["What the satellites show", "Next 7 days", "What the reports tell us", "Suggested actions"]
ARTEFACT = re.compile(r"(DOCUMENT|</?document|This document gives|CoralWatch data:|\s-\s(?:AIMS|Marine Monitoring|Reef Authority|Reef Snapshot|NOAA)\b[^.\[]*$)", re.I)


def tidy_local(text, max_sentences=4):
    """Clean-up for small local models: exact headings, no copied document titles or broken
    fragments, and at most `max_sentences` sentences per section."""
    out, count = [], 0
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            words = _content_words(line)
            best = max(HEADINGS, key=lambda h: len(words & _content_words(h)))
            out.append(f"### {best}")
            count = 0
            continue
        if not line.strip():
            out.append(line)
            continue
        line = ARTEFACT.sub("", line).strip()
        bullet = re.match(r"^\s*([-*]\s+)", line)
        prefix = bullet.group(1) if bullet else ""
        body = line[len(bullet.group(0)):] if bullet else line
        body = re.sub(r"([.!?])\s+((?:\[(?:Data|S\d+)\])+)", r"\1\2", body)
        keep = []
        for sent in re.split(r"(?<=[.!?\]])\s+(?=[A-Z(])", body):
            plain = MARKER.sub("", sent).strip()
            if len(plain.split()) < 4 or not plain[:1].isupper() or ARTEFACT.search(plain):
                continue                                   # fragments like "Park." or mid-sentence copies
            if count >= max_sentences:
                continue
            keep.append(sent)
            count += 1
        if keep:
            out.append(prefix + " ".join(keep))
    return "\n".join(out).strip()
