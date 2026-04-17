"""
clients/pubmed_fetcher.py
=========================
Lightweight NCBI E-utilities client.

Fetches abstracts for a drug query and upserts them into the ChromaStore.
Only triggered when the Chroma collection does not have enough relevant hits.
"""

import re
import time

import requests

from clients.chroma_store import ChromaStore
from utils.constants import (
    ENTREZ_EMAIL,
    ENTREZ_SLEEP,
    ENTREZ_TOOL,
    PUBMED_MAX_FETCH,
)


class PubMedFetcher:
    """
    NCBI E-utilities client for live PubMed abstract retrieval.

    Parameters
    ----------
    store : Shared ChromaStore instance to upsert fetched abstracts into.
    """

    BASE   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    SEARCH = BASE + "/esearch.fcgi"
    FETCH  = BASE + "/efetch.fcgi"
    COMMON = {"tool": ENTREZ_TOOL, "email": ENTREZ_EMAIL, "retmode": "json"}

    def __init__(self, store: ChromaStore):
        self.store = store

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_and_index(self, drug_name: str, max_results: int = PUBMED_MAX_FETCH) -> int:
        """
        Search PubMed for *drug_name*, fetch abstracts, upsert into Chroma.

        Parameters
        ----------
        drug_name   : Drug name to search for.
        max_results : Maximum number of abstracts to retrieve.

        Returns
        -------
        Number of new abstracts indexed.
        """
        print(f"  [PubMed] Live fetch for: {drug_name!r}")
        pmids = self._search(drug_name, max_results)
        if not pmids:
            print(f"  [PubMed] No PMIDs found for {drug_name!r}")
            return 0

        abstracts = self._fetch_abstracts(pmids, drug_name)
        n = self.store.upsert(abstracts)
        print(f"  [PubMed] Upserted {n} abstracts for {drug_name!r}")
        return n

    # ── Internal ──────────────────────────────────────────────────────────────

    def _search(self, drug_name: str, max_results: int) -> list[str]:
        """Run esearch and return a list of PMIDs."""
        query  = f"{drug_name}[MeSH Terms] OR {drug_name}[Title/Abstract]"
        params = {
            **self.COMMON,
            "db": "pubmed", "term": query,
            "retmax": max_results, "usehistory": "n",
        }
        try:
            r = requests.get(self.SEARCH, params=params, timeout=10)
            r.raise_for_status()
            return r.json().get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            print(f"  [PubMed] Search error: {exc}")
            return []

    def _fetch_abstracts(self, pmids: list[str], drug_name: str) -> list[dict]:
        """Run efetch for *pmids* and return parsed abstract dicts."""
        ids_str = ",".join(pmids)
        params  = {
            **self.COMMON,
            "db": "pubmed", "id": ids_str,
            "rettype": "abstract", "retmode": "xml",
        }
        time.sleep(ENTREZ_SLEEP)
        try:
            r = requests.get(self.FETCH, params=params, timeout=30)
            r.raise_for_status()
        except Exception as exc:
            print(f"  [PubMed] Fetch error: {exc}")
            return []

        return self._parse_xml(r.text, drug_name)

    @staticmethod
    def _parse_xml(xml_text: str, drug_name: str) -> list[dict]:
        """Extract PMID, title, and abstract from PubMed XML."""
        articles = []
        for block in re.split(r"<PubmedArticle>", xml_text)[1:]:
            pmid_m  = re.search(r"<PMID[^>]*>(\d+)</PMID>",           block)
            title_m = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", block, re.S)
            abs_m   = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", block, re.S)
            if not (pmid_m and abs_m):
                continue
            articles.append({
                "pmid":     pmid_m.group(1),
                "title":    re.sub(r"<[^>]+>", "", title_m.group(1)) if title_m else "",
                "abstract": re.sub(r"<[^>]+>", "", abs_m.group(1)),
                "drug":     drug_name.lower(),
            })
        return articles