"""Resilient HTTP GET for the external metadata APIs (Scopus, arXiv, Crossref).

Transient failures — read/connection timeouts and 429/5xx — are the dominant
source of run-to-run variation in the pipeline: a dropped *enrichment* call
silently leaves a paper with its un-enriched text, which shifts the BERTopic
corpus and changes the clustering. Retrying so the call reliably completes is
what keeps "same inputs -> same clusters" holding across runs.

One implementation for every caller; each inspects the returned status and
decides what a non-200 means in its context. Only an exhausted transport-level
failure raises.
"""

from __future__ import annotations

import logging
import time
from typing import Mapping, Optional

import requests

DEFAULT_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

logger = logging.getLogger(__name__)


def get_with_retries(
    url: str,
    *,
    params: Optional[Mapping] = None,
    headers: Optional[Mapping] = None,
    timeout: float = 30.0,
    max_attempts: int = 4,
    retry_status=DEFAULT_RETRY_STATUS,
    log: Optional[logging.Logger] = None,
) -> requests.Response:
    """GET ``url`` with exponential backoff on transient failures.

    Retries connection/read timeouts and any status in ``retry_status``
    (honouring a ``Retry-After`` header). Returns the final ``requests.Response``
    once a non-retryable status is reached or attempts are exhausted; raises
    ``requests.RequestException`` only when every attempt fails at the transport
    layer. Pass ``retry_status=frozenset()`` to let the caller handle a status
    itself (e.g. arXiv's 429 sentinel).
    """
    log = log or logger
    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            if attempt >= max_attempts:
                raise
            log.warning(
                "HTTP attempt %d/%d to %s failed (%s); retrying in %.1fs",
                attempt, max_attempts, url, exc, backoff,
            )
            time.sleep(backoff)
            backoff *= 2
            continue
        if resp.status_code in retry_status and attempt < max_attempts:
            wait = backoff
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = max(wait, float(retry_after))
                except ValueError:
                    pass
            log.warning(
                "HTTP %s on attempt %d/%d to %s; retrying in %.1fs",
                resp.status_code, attempt, max_attempts, url, wait,
            )
            time.sleep(wait)
            backoff *= 2
            continue
        return resp
    raise RuntimeError("get_with_retries: unreachable")  # pragma: no cover
