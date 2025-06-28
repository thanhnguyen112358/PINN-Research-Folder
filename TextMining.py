from __future__ import annotations

# ── Standard library
import glob
import os
import re
import sys
from pathlib import Path
from typing import Sequence

# ── Third-party
import numpy as np               # (kept for future use)
import pandas as pd
import PyPDF2
import tiktoken
from dotenv import load_dotenv
from tqdm.auto import tqdm
from openai import OpenAI

# ──────────────────────────────────────────────────────────────
# 0.  Environment / API key
# ──────────────────────────────────────────────────────────────
load_dotenv()                                       # ① load .env if present
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found"

# ──────────────────────────────────────────────────────────────
# 1.  Model & tokenizer
# ──────────────────────────────────────────────────────────────
EMBED_MODEL   = "text-embedding-3-small"
MAX_TOKENS_PER_INPUT = 8191
BATCH_SIZE    = 100

_TOKENIZER = tiktoken.encoding_for_model(EMBED_MODEL)  # build once

# ──────────────────────────────────────────────────────────────
# 2.  Regex helpers
# ──────────────────────────────────────────────────────────────
_HEADER_PAT = re.compile(
    r"\b(?:references|bibliography|works\s+cited|acknowledg(?:e?ments?))\b",
    re.I,
)

_CITE_PAT = re.compile(
    r"""
    ^\s*
    (?:\[\d{1,3}\]|\d{1,3}[.)])     # [12]  or  12.  or  12)
    [\s–\-]*
    .+
    """,
    re.X,
)

# ──────────────────────────────────────────────────────────────
# 3.  Utility functions
# ──────────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    """Exact token count for the active embedding model."""
    return len(_TOKENIZER.encode(text))


def get_pdf_files(folder: Path) -> list[str]:
    pdfs = glob.glob(str(folder / "*.pdf"))
    print(f"Found {len(pdfs)} PDF files in {folder}.")
    return pdfs


def remove_ref(
    text: str,
    *,
    extra_headers: Sequence[str] | None = None,
    min_consecutive: int = 3,
) -> str:
    """Drop ‘References’ / ‘Bibliography’ section heuristically."""
    if not text:
        return text

    hdr_pat = _HEADER_PAT if not extra_headers else re.compile(
        rf"{_HEADER_PAT.pattern}|{'|'.join(map(re.escape, extra_headers))}",
        re.I,
    )
    if (hit := hdr_pat.search(text)):
        return text[: hit.start()].rstrip()

    lines = text.splitlines()
    streak = 0
    for i in range(len(lines) - 1, -1, -1):
        if _CITE_PAT.match(lines[i]):
            streak += 1
            if streak >= min_consecutive:
                return "\n".join(lines[:i]).rstrip()
        else:
            streak = 0
    return text.strip()


def get_txt_from_pdf(
    pdfs: list[str],
    *,
    filter_ref: bool = True,
    min_tokens: int = 40,
) -> pd.DataFrame:
    """Extract quarter-page chunks (≥ min_tokens) from a list of PDFs."""
    rows: list[dict] = []

    for pdf in pdfs:
        try:
            reader = PyPDF2.PdfReader(pdf)
        except Exception as err:
            print(f"⚠️  Skipping {pdf}: {err}")
            continue

        for page_num, page in enumerate(reader.pages, start=1):
            txt = page.extract_text() or ""
            if not txt.strip():
                continue
            if filter_ref:
                txt = remove_ref(txt)

            seg_len = max(len(txt) // 4, 1)
            for idx in range(4):
                part = txt[idx * seg_len : (idx + 1) * seg_len].strip()
                if count_tokens(part) < min_tokens:
                    continue
                rows.append(
                    {
                        "file_name": pdf,
                        "page_number": page_num,
                        "page_section": idx + 1,
                        "content": part,
                        "tokens": count_tokens(part),
                    }
                )

    return pd.DataFrame(rows, dtype="object")

# ──────────────────────────────────────────────────────────────
# 4.  Embedding helper
# ──────────────────────────────────────────────────────────────
def add_embedding(
    df: pd.DataFrame,
    *,
    text_col: str = "content",
    model: str = EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"'{text_col}' column not found")

    client = OpenAI()
    out = df.copy()
    out["embedding"] = pd.Series([None] * len(out), dtype="object")

    texts = out[text_col].astype(str).tolist()
    col_pos = out.columns.get_loc("embedding")          # for .iloc

    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch  = texts[start : start + batch_size]
        resp   = client.embeddings.create(model=model, input=batch)
        embeds = [d.embedding for d in resp.data]

        # --- use .iloc so positions == batch length -----------------
        out.iloc[start : start + len(embeds), col_pos] = embeds
        # ------------------------------------------------------------

        # Equivalent alternative with explicit index alignment:
        # idx = out.index[start : start + len(embeds)]
        # out.loc[idx, "embedding"] = pd.Series(embeds, index=idx)

    return out


# ──────────────────────────────────────────────────────────────
# 5.  Main program
# ──────────────────────────────────────────────────────────────
def main() -> None:
    default_dir = Path(r"E:\PINNs Research\Searched Articles")
    folder      = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir

    pdf_files = get_pdf_files(folder)
    if not pdf_files:
        print("No PDF files found.")
        return

    df  = get_txt_from_pdf(pdf_files)
    df  = add_embedding(df)                    # NEW column added

    print(df.head())

    out_path = folder / "pdf_sections.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()