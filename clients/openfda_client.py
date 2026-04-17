"""
clients/openfda_client.py
=========================
Query the openFDA /drug/event.json endpoint for adverse event counts.

Two counts are returned per drug:
  - total   : All adverse event reports mentioning the drug.
  - serious : Reports where serious=1 (hospitalisation, death, etc.).

Rate limit: 240 requests/minute without an API key (sufficient here).
"""

import requests

from utils.constants import FDA_BASE, FDA_TIMEOUT


class OpenFDAClient:
    """
    Stateless openFDA /drug/event.json client.

    Each call to :meth:`query` fires two HTTP requests and is independent
    of any other call.
    """

    @staticmethod
    def _count(drug_name: str, extra_filter: str = "") -> int:
        """Fire a single count query and return the total."""
        search = f'patient.drug.medicinalproduct:"{drug_name}"'
        if extra_filter:
            search += f" AND {extra_filter}"
        params = {"search": search, "limit": 1}
        try:
            r = requests.get(FDA_BASE, params=params, timeout=FDA_TIMEOUT)
            if r.status_code == 404:
                return 0   # drug not found — not an error
            r.raise_for_status()
            return r.json().get("meta", {}).get("results", {}).get("total", 0)
        except Exception as exc:
            print(f"  [openFDA] Query error for {drug_name!r}: {exc}")
            return 0

    def query(self, drug_name: str) -> tuple[int, int]:
        """
        Return (total_events, serious_events) for *drug_name*.

        Parameters
        ----------
        drug_name : Drug name as it appears in the claim (e.g. "Ibuprofen").

        Returns
        -------
        (total, serious) : Integer counts from openFDA.
        """
        if not drug_name:
            return 0, 0
        total   = self._count(drug_name)
        serious = self._count(drug_name, extra_filter="serious:1")
        return total, serious