from __future__ import annotations

import argparse
import ast
import glob
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import pandas as pd
import PyPDF2
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI  # >= 1.0.0 client
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

# ────────────────────────────────────────────────────────────────────────────
# Environment / client setup
# ────────────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Export it or add to .env.")

client = OpenAI(api_key=API_KEY)

EMBED_MODEL = "text-embedding-3-small"
MAX_TOKENS_PER_INPUT = 8191            # model spec
MAX_TOKENS_PER_BATCH = 300_000         # hard API cap (server side)
SAFETY_FACTOR = 0.9                    # keep 10 % below the cap to avoid drift
EFFECTIVE_BATCH_LIMIT = int(MAX_TOKENS_PER_BATCH * SAFETY_FACTOR)

TOKENS_PER_MINUTE_LIMIT = 1_000_000   # model quota
REQUESTS_PER_MINUTE_LIMIT = 3_000     # model quota

# use model‑specific tokenizer to avoid under‑counting
_TOKENIZER = tiktoken.encoding_for_model(EMBED_MODEL)

# ────────────────────────────────────────────────────────────────────────────
# Regexes pre‑compiled for speed
# ────────────────────────────────────────────────────────────────────────────
_HEADER_PAT = re.compile(
    r"\b(?:references|bibliography|works\s+cited|acknowledg(?:e?ments?))\b",
    re.I,
)

_CITE_PAT = re.compile(
    r"""
    ^\s*                                # leading whitespace
    (?:\[\d{1,3}\]|\d{1,3}[.)])       # [12] | 12. | 12)
    [\s–\-]*                            # optional dash/space
    .+                                   # some body text
    """,
    re.X,
)

# ────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Accurate token count using the tokenizer for ``EMBED_MODEL``."""
    return len(_TOKENIZER.encode(text))


def get_pdf_files(folder: str | Path) -> list[str]:
    folder = Path(folder)
    pdfs = glob.glob(str(folder / "*.pdf"))
    print(f"Found {len(pdfs)} PDF files in {folder}.")
    return pdfs

# ────────────────────────────────────────────────────────────────────────────
# Reference stripping & page‑chunk extraction
# ────────────────────────────────────────────────────────────────────────────

def remove_ref(text: str, *, extra_headers: Sequence[str] | None = None, min_consecutive: int = 3) -> str:
    if not text:
        return text

    hdr_pat = _HEADER_PAT if not extra_headers else re.compile(
        fr"{_HEADER_PAT.pattern}|{'|'.join(map(re.escape, extra_headers))}", re.I
    )
    hit = hdr_pat.search(text)
    if hit:
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


def get_txt_from_pdf(pdfs: list[str], *, filter_ref: bool = False):
    """Yield quarter‑page chunks (≥40 tokens) from a list of PDFs."""
    rows = []
    for pdf in pdfs:
        with open(pdf, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page_num, page in enumerate(reader.pages, start=1):
                txt = page.extract_text() or ""
                if not txt.strip():
                    continue
                if filter_ref:
                    txt = remove_ref(txt)
                # quarter‑page split
                seg_len = max(len(txt) // 4, 1)
                parts = [txt[i * seg_len : (i + 1) * seg_len] for i in range(4)]
                for idx, part in enumerate(parts, start=1):
                    if count_tokens(part) < 40:
                        continue
                    rows.append(
                        {
                            "file name": pdf,
                            "page number": page_num,
                            "page section": idx,
                            "content": part.strip(),
                            "tokens": count_tokens(part),
                        }
                    )

    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────────────────────
# Embedding utilities (token‑aware batching)
# ────────────────────────────────────────────────────────────────────────────

def _iter_token_buckets(texts: list[str]) -> Iterator[list[str]]:
    """Yield batches whose *total* tokens ≤ EFFECTIVE_BATCH_LIMIT."""
    bucket: list[str] = []
    tok_sum = 0
    for txt in texts:
        tks = count_tokens(txt)
        if tks > MAX_TOKENS_PER_INPUT:
            print(
                f"⚠️  Skipping one chunk with {tks} tokens (>{MAX_TOKENS_PER_INPUT} per‑input limit)."
            )
            continue
        if tok_sum + tks > EFFECTIVE_BATCH_LIMIT and bucket:
            yield bucket
            bucket, tok_sum = [txt], tks
        else:
            bucket.append(txt)
            tok_sum += tks
    if bucket:
        yield bucket

@retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(6))
def _embed_once(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)  # type: ignore[arg-type]
    return [d.embedding for d in resp.data]


def add_embeddings(df: pd.DataFrame, *, text_col: str = "content", emb_col: str = "embedding") -> pd.DataFrame:
    df = df.copy()
    if emb_col not in df.columns:
        df[emb_col] = pd.Series(index=df.index, dtype=object)
    else:
    # Column already exists → *ensure* it’s object now
        if df[emb_col].dtype != "object":
            df[emb_col] = df[emb_col].astype(object)
        
    pending_mask = df[emb_col].isna()
    texts = df.loc[pending_mask, text_col].astype(str).tolist()
    if not texts:
        print("✅  Embeddings already present – nothing to do.")
        return df

    # Track which indices will actually get embeddings
    pending_indices = df.loc[pending_mask].index.tolist()
    eligible_texts = []
    eligible_indices = []

    # Filter out texts that are too long before batching
    for idx, text in zip(pending_indices, texts):
        if count_tokens(text) > MAX_TOKENS_PER_INPUT:
            print(f"⚠️  Skipping chunk with {count_tokens(text)} tokens (>{MAX_TOKENS_PER_INPUT} limit)")
            continue
        eligible_texts.append(text)
        eligible_indices.append(idx)

    embeds: list[list[float]] = []
    req_count = 0
    tokens_sent_this_minute = 0
    minute_start = time.time()

    for bucket in tqdm(list(_iter_token_buckets(eligible_texts)), desc="Embedding", unit="batch"):
        bucket_tokens = sum(count_tokens(t) for t in bucket)
        embeds.extend(_embed_once(bucket))

        # pacing for TPM / RPM
        req_count += 1
        tokens_sent_this_minute += bucket_tokens
        elapsed = time.time() - minute_start
        if elapsed < 60:
            if tokens_sent_this_minute > TOKENS_PER_MINUTE_LIMIT or req_count >= REQUESTS_PER_MINUTE_LIMIT:
                time.sleep(60 - elapsed)
                minute_start = time.time()
                req_count = 0
                tokens_sent_this_minute = 0
        else:
            minute_start = time.time()
            req_count = 0
            tokens_sent_this_minute = 0

    # Only update rows that we actually processed
    for idx, embed in zip(eligible_indices, embeds):
        df.at[idx, emb_col] = np.asarray(embed, dtype=np.float32)
    
    return df

# ────────────────────────────────────────────────────────────────────────────
# Similarity helpers
# ────────────────────────────────────────────────────────────────────────────

def add_similarity(df: pd.DataFrame, vec: list[float]):
    df = df.copy()
    df["similarity"] = df["embedding"].apply(
        lambda e: cosine_similarity([e], [vec])[0][0] if isinstance(e, (list, tuple, np.ndarray)) else np.nan
    )
    return df


def select_top_neighbors(df: pd.DataFrame, k: int = 10):
    """Return top‑k chunks per file, plus adjacent chunks for context."""
    df = df.sort_values(["file name", "similarity"], ascending=[True, False])
    topk = df.groupby("file name").head(k)
    neigh_idx = [i for idx in topk.index for i in (idx - 1, idx + 1) if 0 <= i < len(df)]
    return df.loc[topk.index.union(neigh_idx)]

# ---------------------------------------------------------------------------
# CSV helpers ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def df_to_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, escapechar="\\")


def csv_to_df(path: str):
    """Load CSV and convert any stringified embeddings back to list[float]."""
    df = pd.read_csv(path)
    if "embedding" in df.columns:
        df["embedding"] = df["embedding"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
        )
    return df

# # ---------------------------------------------------------------------------
# # Model functions -----------------------------------------------------------
# # ---------------------------------------------------------------------------

# @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(6))
# def _query_model(prompt: str) -> str:
#     """Send a prompt to the model and return the response with retry logic."""
#     try:
#         response = client.chat.completions.create(
#             model="o3",  # or appropriate model
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         print(f"Error querying model: {e}")
#         raise

# # ---------------------------------------------------------------------------
# # Model implementations ------------------------------------------------------
# # ---------------------------------------------------------------------------

# def Model_1(df):
#     """
#     Process DataFrame content through the OpenAI API.
    
#     Args:
#         df: DataFrame with 'content' column
        
#     Returns:
#         DataFrame with added 'summarized' column containing model responses
#     """
#     # Make a copy to avoid modifying the input
#     result_df = df.copy()
    
#     # Define the tubulation prompt
#     tubulation_prompt = "\n\nGiven the above context, please extract and summarize any experimental conditions, parameters, or methodology details related to chemical looping gasification for hydrogen production."
    
#     # Process each row
#     summaries = []
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing content"):
#         content = row['content']
#         token_count = count_tokens(content)
        
#         # If content exceeds token limit, split it
#         if token_count > 3000:
#             # Calculate how many chunks we need
#             num_chunks = math.ceil(token_count / 3000)
#             # Approximate character length per chunk
#             chars_per_chunk = len(content) // num_chunks
            
#             # Split content into chunks
#             chunks = [content[i*chars_per_chunk:(i+1)*chars_per_chunk] for i in range(num_chunks)]
            
#             # Process each chunk and combine results
#             chunk_results = []
#             for chunk in chunks:
#                 prompt = f"Context: {chunk}{tubulation_prompt}"
#                 chunk_answer = _query_model(prompt)
#                 chunk_results.append(chunk_answer)
                
#             # Combine chunk results
#             answer = "\n\n".join(chunk_results)
            
#         else:
#             # Process content as a single prompt
#             prompt = f"Context: {content}{tubulation_prompt}"
#             answer = _query_model(prompt)
        
#         summaries.append(answer)
    
#     # Add summaries to dataframe
#     result_df["summarized"] = summaries
    
#     return result_df

# def Model_2(
#     df: pd.DataFrame,
#     model_id: str = "o3-mini",
#     sleep_s: float = 0.25,
# ) -> pd.DataFrame:
#     """
#     Binary-classify passages as containing >=2 quantitative CLG experimental parameters.
#     """

#     SYSTEM_MSG = (
#         "You are a classifier. You must answer with exactly 'Yes' or 'No'. "
#         "Answer 'Yes' only if the text contains NUMERICAL values for AT LEAST TWO of "
#         "these categories: (1) operating temperature, (2) steam/biomass or O2/fuel ratio "
#         "or pressure, (3) quantity of fuel or oxygen carrier, (4) reaction or residence time. "
#         "Otherwise, answer 'No'."
#     )

#     FEW_SHOT = """
# Example Yes 1:
# The gasification was performed at 850 °C with a steam/biomass ratio of 1.5.
# Answer: Yes

# Example Yes 2:
# A pine-sawdust feed (10 g) was gasified with 50 g Fe₂O₃/Al₂O₃ at 900 °C for 15 min.
# Answer: Yes

# Example Yes 3:
# Experiments at 5 bar used 250 g ilmenite (OC/Fuel = 5 wt %) and 20 g coal.
# Answer: Yes

# Example No 1:
# Chemical looping gasification is a promising low-carbon route for hydrogen.
# Answer: No

# Example No 2:
# Previous studies examined temperatures between 800 and 950 °C.
# Answer: No

# Example No 3:
# Hydrogen yield reached 55 vol % at optimal conditions.
# Answer: No
# """.strip()

#     df_out = df.copy()
#     answers = []

#     for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying (v2)"):
#         prompt = (
#             f"{FEW_SHOT}\n\nText: {row['content']}\n\n"
#             "Question: Does the text meet the two-parameter rule? Answer:"
#         )

#         time.sleep(sleep_s)
#         raw = _query_model(
#             messages=[
#                 {"role": "system", "content": SYSTEM_MSG},
#                 {"role": "user",   "content": prompt},
#             ],
#             model=model_id,            # pass through
#         )
#         first = raw.strip().split()[0].lower()
#         answers.append("yes" if first.startswith("y") else "no")

#     df_out["classification"] = answers
#     print(answers.count("yes"), "positive classifications found.")
    
#     positives = df_out[df_out["classification"] == "yes"]
#     return Model_1(positives)


# def Model_3(
#     df: pd.DataFrame,
#     prompt_choice: str = "clg_conditions",
#     run_classification: bool = True,
#     embedding_model: str = "text-embedding-3-small",
# ):
#     """
#     Stage-1 retrieval for CLG/H2 domain.
#     1) Embed the domain prompt, 2) compute cosine sim for each chunk,
#     3) keep top-k chunks + neighbours, 4) optionally call Model_2.
#     """

#     # ---- 1. prompt selection ------------------------------------------------
#     clg_prompts = {
#     "clg_conditions": (
#         "Identify passages that detail experimental conditions for chemical looping "
#         "gasification aimed at hydrogen-rich syngas. Include: biomass or coal type and amount "
#         "(e.g., 10 g pine sawdust), oxygen-carrier composition and loading "
#         "(e.g., 50 g Fe₂O₃/Al₂O₃), reactor configuration (dual fluidised bed, batch), "
#         "operating temperature (820–930 °C), pressure, steam-to-biomass ratio, "
#         "oxygen-to-fuel lambda, residence time, particle size, carrier regeneration steps."
#     ),
#     "clg_performance": (
#         "Locate sections reporting CLG performance metrics: syngas hydrogen fraction, "
#         "H₂/CO ratio, cold-gas efficiency, carbon conversion, tar yield, and oxygen-carrier attrition."
#     ),
#     "oc_characterisation": (
#         "Locate passages that characterise the oxygen carrier used in CLG: BET surface area, "
#         "crushing strength, XRD phases after redox cycling, TGA weight loss, SEM morphology."
#     )
# }

#     prompt = clg_prompts.get(
#         prompt_choice,
#         f"Locate passages discussing '{prompt_choice}' in chemical looping gasification."
#     )

#     # ---- 2. embed the prompt -----------------------------------------------
#     prompt_emb = client.embeddings.create(
#         model=embedding_model,
#         input=prompt
#     ).data[0].embedding

#     # ---- 3. make sure chunks have embeddings -------------------------------
#     if "embedding" not in df.columns:
#         df = add_embeddings(df, model=embedding_model)

#     # ---- 4. similarity + neighbour expansion -------------------------------
#     df_sim = add_similarity(df, prompt_emb)
#     df_top = select_top_neighbors(df_sim, k=10, neighbour_window=1)

#     # ---- 5. optional Yes/No gate -------------------------------------------
#     return Model_2(df_top) if run_classification else df_top

# ---------------------------------------------------------------------------
# Main entry -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    import argparse, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?",
                        default=r"E:\PINNs Research\Searched Articles",
                        help="Folder with PDFs")
    parser.add_argument("--model", default="o3",
                        help="ChatCompletion model id (e.g. o3, o3-mini)")
    args = parser.parse_args()

    pdf_files = get_pdf_files(args.directory)

    # 1) PDF → dataframe
    df_raw = get_txt_from_pdf(pdf_files, filter_ref=True)

    # 2) Embed once up front
    df_emb = add_embeddings(df_raw)
    print("✅  Added embeddings to DataFrame.")

    # (optional) checkpoint before costly LLM calls
    csv_emb = os.path.join(args.directory, "pdf_chunks_with_embeddings.csv")
    df_to_csv(df_emb, csv_emb)
    print(f"✅  Saved embeddings to {csv_emb}")

    # 3) Run the full LLM pipeline
    # df_final = Model_3(df_emb)

    # # 4) Persist final results
    # csv_final = os.path.join(args.directory, "mof_extractions.csv")
    # df_final.to_csv(csv_final, index=False)
    # print(f"🎉  Pipeline complete! Results saved to {csv_final}")


if __name__ == "__main__":
    main()
