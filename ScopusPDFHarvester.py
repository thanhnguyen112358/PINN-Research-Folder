
"""Scopus Search‑API PDF harvester
=================================
* Replaces WoS Starter with Elsevier Scopus Search API v2.
* Harvests up to `max_results` records for a query; extracts title + DOI.
* Persists a CSV and downloads OA PDFs via Unpaywall.

Environment variables expected
------------------------------
SCOPUS_API_KEY   – Elsevier developer key (mandatory)
UNPAYWALL_EMAIL  – email for Unpaywall requests (defaults to placeholder)

Usage
-----
python scopus_pdf_harvester.py --query "TITLE(biomass AND gasification)" --max 100
"""

from __future__ import annotations

import os
import time
import csv
import logging
import argparse
from dataclasses import dataclass
from typing import List, Optional

import requests

import openai, backoff, os, re

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

openai.api_key = "***"

_SYSTEM_PROMPT = (
    "You are an expert in biomass gasification. "
    "Answer only with 'yes' or 'no'. "
    "Reply 'yes' **only** if the abstract describes research that "
    "focuses on hydrogen production via *chemical-looping gasification* "
    "OR explicitly uses a database of CLG experiments/models."
)

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60)
def is_relevant(abstract: str) -> bool:
    msg = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": abstract.strip()}
    ]
    rsp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # cheap & fast; bump up if accuracy suffers
        messages=msg,
        temperature=0,
        max_tokens=1
    )
    return rsp.choices[0].message.content.lower().startswith("y")

# -----------------------------------------------------------------------------

SCOPUS_API_KEY = "ed015278de4e237401c8db1c63ec821a"
#print(f"SCOPUS_API_KEY: {SCOPUS_API_KEY}")
if not SCOPUS_API_KEY:
    raise RuntimeError(
        "SCOPUS_API_KEY not set. Obtain a key at https://dev.elsevier.com and export it first."
    )

SCOPUS_ENDPOINT = "https://api.elsevier.com/content/search/scopus"
ABSTRACT_ENDPOINT = "https://api.elsevier.com/content/abstract/eid/{}"
UNPAYWALL_EMAIL = "dn018@bucknell.edu"
DEFAULT_MAX_RESULTS = 200 # Default for testing; set higher in production
PAGE_SIZE = 25  # Scopus allows up to 200; choose 25 to be polite

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class Paper:
    title: str
    doi: Optional[str]

# -----------------------------------------------------------------------------
# Scopus Search wrapper
# -----------------------------------------------------------------------------

def scopus_search(query: str, start: int, count: int,
                  *, view: str = "STANDARD") -> dict:
    headers = {
        "X-ELS-APIKey": SCOPUS_API_KEY,
        "Accept": "application/json",
        "User-Agent": "ScopusPDFHarvester/1.2 (+https://github.com/yourrepo)"
    }
    params = {"query": query, "start": start, "count": count, "view": view}

    resp = requests.get(SCOPUS_ENDPOINT, headers=headers,
                        params=params, timeout=30)

    # If COMPLETE fails, retry once with STANDARD
    if resp.status_code == 401 and view.upper() == "COMPLETE":
        logging.info("COMPLETE view not authorised – retrying with STANDARD")
        params["view"] = "STANDARD"
        resp = requests.get(SCOPUS_ENDPOINT, headers=headers,
                            params=params, timeout=30)

    resp.raise_for_status()
    return resp.json()

def from_crossref(doi):
    if not doi: return None
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}",
                         params={"mailto": "dn018@bucknell.edu"}, timeout=15)
        r.raise_for_status()
        raw = r.json()["message"].get("abstract")
        return re.sub("<[^>]+>", "", raw) if raw else None
    except requests.RequestException as exc:
        logging.warning("Crossref failed %s – %s", doi, exc)

def from_semanticscholar(doi):
    if not doi: return None
    try:
        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{doi}",
            params={"fields": "abstract"}, headers={"User-Agent": "Harvester/1.0"},
            timeout=15)
        r.raise_for_status()
        return r.json().get("abstract")
    except requests.RequestException as exc:
        logging.warning("S2 failed %s – %s", doi, exc)

def from_lens(doi):
    TOKEN = os.getenv("LENS_TOKEN")
    if not (doi and TOKEN): return None
    try:
        r = requests.get("https://api.lens.org/scholarlyworks",
                         params={"doi": doi, "fields": "abstract"},
                         headers={"Authorization": f"Bearer {TOKEN}"}, timeout=15)
        r.raise_for_status()
        data = r.json().get("data")
        return data[0].get("abstract") if data else None
    except requests.RequestException as exc:
        logging.warning("Lens failed %s – %s", doi, exc)

def from_openalex(doi):
    if not doi: return None
    try:
        r = requests.get(f"https://api.openalex.org/works/doi:{doi}", timeout=15)
        r.raise_for_status()
        idx = r.json().get("abstract_inverted_index")
        if not idx: return None
        words = sorted((p, w) for w, poses in idx.items() for p in poses)
        return " ".join(w for p, w in words)
    except requests.RequestException as exc:
        logging.warning("OpenAlex failed %s – %s", doi, exc)

def fetch_abstract(doi):
    for fn in (from_crossref, from_semanticscholar, from_lens, from_openalex):
        text = fn(doi)
        if text: return text
    return None

# -----------------------------------------------------------------------------
# Fetch abstract via DOI
# -----------------------------------------------------------------------------

def fetch_abstract(doi: str | None) -> str | None:
    for fn in (from_crossref, from_semanticscholar,
               from_lens, from_openalex):
        text = fn(doi)
        if text:
            return text
    return None

# -----------------------------------------------------------------------------
# Collect title + DOI until max_results
# -----------------------------------------------------------------------------

def get_filtered_papers(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[Paper]:
    accepted, start = [], 0
    while len(accepted) < max_results:
        remaining = max_results - len(accepted) 
        count = min(PAGE_SIZE, remaining)
        data = scopus_search(query, start=start, count=count, view="STANDARD")
        
        print (data)
        
        for entry in data.get("search-results", {}).get("entry", []):
            title = entry.get("dc:title") or entry.get("prism:title") or "<no title>"
            doi   = entry.get("prism:doi")
            abstract = fetch_abstract(doi)            # ← now implemented
            if abstract and is_relevant(abstract):
                accepted.append(Paper(title=title, doi=doi))
                if len(accepted) >= max_results:
                    break
        start += count
        if not data.get("search-results", {}).get("entry"):
            break
    return accepted


# -----------------------------------------------------------------------------
# Unpaywall PDF downloader (unchanged from WoS version, but slugifies filenames)
# -----------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Return a filesystem‑safe slug."""
    import re
    slug = re.sub(r"[^\w-]", "_", text)
    return slug[:255]


def download_pdf(doi: Optional[str], *, save_dir: str = "pdfs") -> bool:
    if not doi:
        logging.info("No DOI – skipping download.")
        return False

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{slugify(doi)}.pdf"
    path = os.path.join(save_dir, filename)

    # 1️⃣ Try Unpaywall
    try:
        upw_resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}",
            timeout=20,
        )
        upw_resp.raise_for_status()
        upw_data = upw_resp.json()
        loc = upw_data.get("best_oa_location") or next(iter(upw_data.get("oa_locations") or []), {})
        pdf_url = loc.get("url_for_pdf")
        if pdf_url:
            pdf_resp = requests.get(pdf_url, timeout=90)
            pdf_resp.raise_for_status()
            if pdf_resp.content[:4] == b"%PDF":
                with open(path, "wb") as fh:
                    fh.write(pdf_resp.content)
                logging.info("Saved OA PDF → %s", path)
                return True
            else:
                logging.info("URL did not return a valid PDF for %s", doi)
    except requests.exceptions.RequestException as exc:
        logging.warning("OA PDF failed for %s – %s", doi, exc)

    # 2️⃣ Fallback to Sci-Hub
    try:
        scihub_url = f"https://sci-hub.se/{doi}"
        resp = requests.get(scihub_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.content, 'html.parser')
        iframe = soup.find('iframe')
        if iframe:
            pdf_url = iframe['src']
            if pdf_url.startswith('//'):
                pdf_url = 'https:' + pdf_url
            elif pdf_url.startswith('/'):
                pdf_url = 'https://sci-hub.se' + pdf_url

            pdf_resp = requests.get(pdf_url, timeout=90, headers={'User-Agent': 'Mozilla/5.0'})
            pdf_resp.raise_for_status()

            if pdf_resp.content[:4] == b"%PDF":
                with open(path, "wb") as fh:
                    fh.write(pdf_resp.content)
                logging.info("Saved Sci-Hub PDF → %s", path)
                return True
            else:
                logging.info("Sci-Hub URL did not return valid PDF for %s", doi)
        else:
            logging.info("No Sci-Hub iframe found for %s", doi)
    except requests.exceptions.RequestException as exc:
        logging.warning("Sci-Hub failed for %s – %s", doi, exc)

    logging.info("Failed to download PDF for %s", doi)
    return False

# -----------------------------------------------------------------------------
# CSV helper
# -----------------------------------------------------------------------------

def save_csv(rows: List[Paper], *, path: str = "scopus_results.csv") -> None:
    if not rows:
        logging.info("No records – skip CSV.")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["title", "doi"])
        for p in rows:
            writer.writerow([p.title, p.doi or ""])
    logging.info("Wrote CSV → %s", path)

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest Scopus titles + OA PDFs via Unpaywall.")
    parser.add_argument("--query", default="""TITLE-ABS-KEY(
      (Hydrogen) AND (chemical-looping gasification) )""", 
      help="Scopus query string (API syntax)")
    parser.add_argument("--max", type=int, default=200, help="Maximum results to harvest")
    parser.add_argument("--csv", default="scopus_results_HCLG_4o.csv", help="CSV path to write metadata")
    parser.add_argument("--pdf-dir", default="downloaded_pdfs_HCLG_4o", help="Directory to save PDFs")
    args = parser.parse_args()

    papers = get_filtered_papers(args.query, max_results=args.max)
    save_csv(papers, path=args.csv)

    for p in papers:
        logging.info("TITLE: %s", p.title)
        download_pdf(p.doi, save_dir=args.pdf_dir)
        time.sleep(1.1)  # polite Unpaywall delay

if __name__ == "__main__":
    main()