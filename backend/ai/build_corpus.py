"""
build_corpus.py
===============
Turns the reef reports into a searchable passage corpus for the LLM explainer.

    python -m backend.ai.build_corpus

Reads the PDFs straight from NLP/nlp_dataset/*.zip (AIMS, MMP, Reef Snapshot),
plus a few reference web pages, and writes backend/data/corpus.json:

    [{"id", "source", "title", "published", "page", "text"}, ...]

"published" is the year the report came out (e.g. an AIMS 2023-2024 report -> 2024).
The explainer only uses passages published up to the year being explained, so an
explanation of March 2022 can't quote a report written after it.
"""

import io
import json
import re
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
ZIP_DIR = ROOT / "NLP" / "nlp_dataset"
OUT = Path(__file__).resolve().parents[1] / "data" / "corpus.json"

CHUNK_WORDS = 220
OVERLAP_WORDS = 40
MIN_WORDS = 40

SOURCE_NAMES = {
    "AIMS": "AIMS Long-Term Monitoring Program",
    "MMP": "Marine Monitoring Program",
    "Reef": "Reef Snapshot / Reef health updates",
}

# Short reference pages that explain the satellite products. Fetched at build time;
# any that fail are skipped.
WEB_PAGES = [
    ("NOAA Coral Reef Watch", "Coral Reef Watch 5 km Degree Heating Week product",
     "https://coralreefwatch.noaa.gov/product/5km/index_5km_dhw.php"),
    ("NOAA Coral Reef Watch", "Coral Reef Watch 5 km Bleaching Alert Area product",
     "https://coralreefwatch.noaa.gov/product/5km/index_5km_baa-max-7d.php"),
    ("NOAA Coral Reef Watch", "Coral Reef Watch 5 km HotSpot product",
     "https://coralreefwatch.noaa.gov/product/5km/index_5km_hs.php"),
]


def published_year(name):
    """Latest 4-digit year in the file name, expanding '2019-20' / '2019_20' style ranges."""
    years = [int(y) for y in re.findall(r"20\d{2}", name)]
    short = re.search(r"(20\d{2})[-_](\d{2})(?!\d)", name)
    if short:
        years.append(int(short.group(1)[:2] + short.group(2)))
    return max(years) if years else None


def clean(text):
    text = re.sub(r"-\n(\w)", r"\1", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)  # page numbers on their own line
    return re.sub(r"\s*\n\s*", " ", text).strip()


def chunk_pages(pages):
    """Split [(page_no, text)] into overlapping word windows that remember their start page."""
    words, owners = [], []
    for page_no, text in pages:
        w = text.split()
        words += w
        owners += [page_no] * len(w)
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, max(1, len(words) - OVERLAP_WORDS), step):
        piece = words[start:start + CHUNK_WORDS]
        if len(piece) >= MIN_WORDS:
            yield owners[start], " ".join(piece)


def pdf_passages():
    for zpath in sorted(ZIP_DIR.glob("*.zip")):
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                if not info.filename.lower().endswith(".pdf"):
                    continue
                name = Path(info.filename).name
                folder = info.filename.split("/")[0]
                source = "AIMS" if folder.startswith("AIMS") else "MMP" if folder.startswith("MMP") else "Reef"
                try:
                    reader = PdfReader(io.BytesIO(z.read(info)))
                    pages = [(i + 1, clean(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
                except Exception as e:
                    print(f"  skip {name}: {e}")
                    continue
                pages = [(n, t) for n, t in pages if len(t.split()) > 20]
                if not pages:
                    print(f"  skip {name}: no text layer (scanned PDF)")
                    continue
                year = published_year(name)
                n = 0
                for page, text in chunk_pages(pages):
                    n += 1
                    yield {"source": SOURCE_NAMES[source], "title": name.rsplit(".", 1)[0],
                           "published": year, "page": page, "text": text}
                print(f"  {name}: {len(pages)} pages -> {n} passages (published {year})")


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "noscript"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        self._skip += tag in self.SKIP

    def handle_endtag(self, tag):
        self._skip -= tag in self.SKIP and self._skip > 0

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def web_passages():
    for source, title, url in WEB_PAGES:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CoralWatch-capstone/1.0"})
            html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
        except Exception as e:
            print(f"  skip {url}: {e}")
            continue
        p = _TextExtractor()
        p.feed(html)
        text = clean(" ".join(p.parts))
        n = 0
        for _, piece in chunk_pages([(1, text)]):
            n += 1
            yield {"source": source, "title": title, "published": None, "page": None, "text": piece, "url": url}
        print(f"  {title}: {n} passages")


def date_undated(passages):
    """Reef health updates bundle many dated notes in one PDF with no year in the file
    name. Date each passage by the latest year it mentions, else by its neighbour."""
    prev = {}
    for p in passages:
        if p["published"] is not None or p.get("url"):
            continue
        years = [int(y) for y in re.findall(r"\b(20[12]\d)\b", p["text"]) if 2014 <= int(y) <= 2026]
        p["published"] = max(years) if years else prev.get(p["title"])
        prev[p["title"]] = p["published"]


def main():
    print("Reading reef report PDFs ...")
    passages = list(pdf_passages())
    date_undated(passages)
    print("Fetching reference web pages ...")
    passages += list(web_passages())
    for i, p in enumerate(passages):
        p["id"] = i
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(passages))
    print(f"Saved {OUT.relative_to(ROOT)} ({len(passages)} passages)")


if __name__ == "__main__":
    main()
