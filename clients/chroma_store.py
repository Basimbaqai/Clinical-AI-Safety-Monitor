"""
clients/chroma_store.py
=======================
Thin wrapper around a persistent Chroma collection that stores
PubMed abstracts as 768-d S-PubMedBERT embeddings.
"""

from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from models.evidence import SearchResult
from utils.constants import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    TOP_K,
)


class ChromaStore:
    """
    Manages a persistent Chroma collection for semantic search over PubMed abstracts.

    The embedding model is loaded once and reused for both indexing and querying.

    Parameters
    ----------
    embed_model : SentenceTransformer model name for 768-d embeddings.
    chroma_path : Filesystem path for the persisted Chroma store.
    collection  : Name of the Chroma collection to use.
    """

    def __init__(
        self,
        embed_model: str = EMBED_MODEL_NAME,
        chroma_path: str = CHROMA_PATH,
        collection:  str = COLLECTION_NAME,
    ):
        print(f"[ChromaStore] Loading embedding model: {embed_model}")
        self.embedder = SentenceTransformer(embed_model)

        print(f"[ChromaStore] Opening Chroma store at: {chroma_path}")
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.col = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[ChromaStore] Collection '{collection}' — {self.col.count()} docs")

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, text: str) -> list[float]:
        """Return a 768-d embedding for *text*."""
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        claim_text:  str,
        top_k:       int            = TOP_K,
        drug_filter: Optional[str]  = None,
    ) -> list[SearchResult]:
        """
        Semantic search: encode the claim and return the top-K closest abstracts.

        Parameters
        ----------
        claim_text  : Raw claim sentence.
        top_k       : Number of results to retrieve.
        drug_filter : If provided, restrict results to documents tagged with
                      this drug name (stored in the ``drug`` metadata field).

        Returns
        -------
        List of SearchResult objects ordered by cosine similarity.
        """
        vector = self.encode(claim_text)
        where  = {"drug": {"$eq": drug_filter.lower()}} if drug_filter else None

        try:
            results = self.col.query(
                query_embeddings=[vector],
                n_results=min(top_k, max(self.col.count(), 1)),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Fall back without drug filter if collection is empty or filter fails
            results = self.col.query(
                query_embeddings=[vector],
                n_results=min(top_k, max(self.col.count(), 1)),
                include=["documents", "metadatas", "distances"],
            )

        hits:      list[SearchResult] = []
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            hits.append(SearchResult(
                pmid        = meta.get("pmid", "unknown"),
                abstract    = doc,
                title       = meta.get("title", ""),
                distance    = float(dist),
                drug_filter = drug_filter or "",
            ))
        return hits

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert(self, abstracts: list[dict]) -> int:
        """
        Add or update documents in the collection.

        Each item in *abstracts* must contain:
          - ``pmid``     : str
          - ``abstract`` : str
          - ``title``    : str  (optional)
          - ``drug``     : str  (optional, lower-cased for filtering)

        Returns the number of documents upserted.
        """
        if not abstracts:
            return 0

        ids, docs, metas, embeds = [], [], [], []
        for item in abstracts:
            pmid = str(item["pmid"])
            text = item.get("abstract", "")
            if not text.strip():
                continue
            ids.append(pmid)
            docs.append(text)
            metas.append({
                "pmid":  pmid,
                "title": item.get("title", ""),
                "drug":  item.get("drug",  "").lower(),
            })
            embeds.append(self.encode(text))

        if ids:
            self.col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
        return len(ids)