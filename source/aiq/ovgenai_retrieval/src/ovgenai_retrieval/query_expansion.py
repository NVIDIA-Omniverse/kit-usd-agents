"""Glossary-based query expansion (REQ-QE-1 through REQ-QE-6).

Loads ``glossaries/*.yaml`` at first use, indexes them by category +
abbreviation, and exposes :func:`expand` for use by ``HybridRetriever``.
Works purely at query-string augmentation level: the original tokens
are preserved, expansion terms are appended in parentheses so both
semantic and lexical retrievers see them.

Domain-aware design (Phase 3 of the glossary fix proposal):

- The YAML's top-level keys are category names (``usd_modules``,
  ``physics``, ``rendering``, …). They are first-class at runtime — each
  abbreviation entry is tagged with the category it came from.
- Callers pass ``domains={"usd_modules", "geometry"}`` to scope expansion
  to that subset; abbreviations whose entries fall entirely outside the
  set are not expanded.
- When ``domains=None``, every category is eligible (legacy behaviour).
- For abbreviations that occur in multiple eligible categories with
  different expansions (e.g. ``SDF``: ``usd_modules`` vs ``physics``),
  disambiguation is **glossary-derived** — *not* a hard-coded table.
  We score each candidate by counting how many other tokens in the same
  query belong to the same category. The highest-scoring category wins;
  zero-or-tied scores result in no expansion (conservative — the
  existing ``SDF`` collisions today inject *both* expansions, which
  pollutes every downstream retriever).

Logging: expansions are logged at INFO level when the ``OVAI_QE_LOG``
env var is set to a truthy value (``1``, ``true``, ``yes``). REQ-QE-6.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Glossary loading — preserves category structure.
# ---------------------------------------------------------------------------

_GLOSSARIES_DIR = Path(__file__).parent / "glossaries"

# Legacy in-file seed (used only if pyyaml missing AND no YAML found).
# Kept under a synthetic ``_seed`` domain so the new domain-aware code
# path treats it identically to YAML-loaded entries.
_SEED_GLOSSARY: dict[str, list[str]] = {
    "LIVRPS": ["local", "inherits", "variantSets", "references", "payloads", "specializes"],
    "KAT": ["kit-app-template"],
    "UI": ["omni.ui", "user interface"],
    "USD": ["OpenUSD", "Universal Scene Description"],
    "MCP": ["Model Context Protocol"],
}


@dataclass(frozen=True)
class _Entry:
    """One glossary entry, tagged with the category it came from.

    A given ``abbrev`` may have several ``_Entry`` records — one per
    category that defines it (e.g. ``SDF`` has one entry under
    ``usd_modules`` and another under ``physics``).
    """

    domain: str
    terms: tuple[str, ...]


# Module-level meta keys we should never index as abbreviations.
_RESERVED_META_KEYS = {"_meta"}


@lru_cache(maxsize=1)
def _load_glossary_index() -> dict[str, list[_Entry]]:
    """Build ``{ABBREV: [_Entry, ...]}`` from every YAML under glossaries/.

    Keys are uppercased to drive case-insensitive lookup; expansion text
    keeps original casing.
    """
    index: dict[str, list[_Entry]] = {}

    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning("pyyaml not installed; falling back to seed glossary")
        for k, v in _SEED_GLOSSARY.items():
            index.setdefault(k.upper(), []).append(_Entry("_seed", tuple(v)))
        return index

    if not _GLOSSARIES_DIR.is_dir():
        for k, v in _SEED_GLOSSARY.items():
            index.setdefault(k.upper(), []).append(_Entry("_seed", tuple(v)))
        return index

    for path in sorted(_GLOSSARIES_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning(f"query_expansion: skipping {path.name}: {e!r}")
            continue
        for key, value in data.items():
            if str(key) in _RESERVED_META_KEYS:
                continue
            if isinstance(value, list):
                # Flat shape: {abbrev: [terms]} → synthetic ``_uncategorised`` domain.
                index.setdefault(str(key).upper(), []).append(_Entry("_uncategorised", tuple(str(t) for t in value)))
            elif isinstance(value, dict):
                # Nested shape: {category: {abbrev: [terms]}}
                for abbrev, terms in value.items():
                    if isinstance(terms, list):
                        index.setdefault(str(abbrev).upper(), []).append(_Entry(str(key), tuple(str(t) for t in terms)))

    if not index:
        for k, v in _SEED_GLOSSARY.items():
            index.setdefault(k.upper(), []).append(_Entry("_seed", tuple(v)))
    return index


@lru_cache(maxsize=1)
def _domain_token_index() -> dict[str, set[str]]:
    """Inverse of the glossary: ``{DOMAIN: {ABBREV, ...}}``.

    Used to derive context for disambiguation. When a query token
    upper-cases to one of these abbreviations, we know which categories
    that token "belongs to" — useful for picking which expansion of an
    ambiguous abbreviation to apply.
    """
    by_domain: dict[str, set[str]] = {}
    for abbrev, entries in _load_glossary_index().items():
        for e in entries:
            by_domain.setdefault(e.domain, set()).add(abbrev)
    return by_domain


def reset_glossary_cache() -> None:
    """Force the glossary index to rebuild from disk on next call.

    Useful for tests that swap glossary content at runtime.
    """
    _load_glossary_index.cache_clear()
    _domain_token_index.cache_clear()


# Backwards-compat alias — the previous module name used by the test suite.
clear_cache = reset_glossary_cache


def _log_enabled() -> bool:
    return os.environ.get("OVAI_QE_LOG", "").strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

# Token boundary used for whole-word matching. ``[\w.]+`` lets us treat
# ``omni.ui`` as one token (important because many dotted IDs are legitimate
# query tokens).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-]+")


def _filter_eligible(entries: list[_Entry], domains: set[str] | None) -> list[_Entry]:
    """Restrict ``entries`` to those whose ``.domain`` is in ``domains``.

    ``domains=None`` keeps everything. ``_seed`` and ``_uncategorised``
    synthetic domains are always eligible — they predate the category
    feature and shouldn't be silently dropped.
    """
    if domains is None:
        return entries
    allowed = set(domains) | {"_seed", "_uncategorised"}
    return [e for e in entries if e.domain in allowed]


def _disambiguate(
    abbrev: str,
    eligible: list[_Entry],
    query_token_set: set[str],
) -> _Entry | None:
    """Pick one entry from ``eligible`` based on the query's other tokens.

    Algorithm:
        1. If every eligible entry belongs to the **same** glossary
           domain (e.g. ``Gf`` has 3 entries — all from ``usd_modules``,
           one per loaded YAML), there is no real ambiguity. Return any
           one of them.
        2. Otherwise score each candidate by counting how many *other*
           tokens in the query share its domain — but **excluding**
           tokens that are themselves ambiguous, so two unrelated
           multi-domain abbrevs can't mutually reinforce. The highest
           single-best score wins; zero-everywhere or a tie returns
           ``None`` (caller should not expand).

    Fully glossary-derived — no hard-coded keyword tables. If the
    glossary grows new categories, this just works.
    """
    if not eligible:
        return None

    distinct_domains = {e.domain for e in eligible}
    if len(distinct_domains) == 1:
        return eligible[0]

    index = _load_glossary_index()
    domain_index = _domain_token_index()

    # Other query tokens that disambiguate cleanly (i.e. live in just
    # one domain themselves). An ambiguous co-occurring abbrev like
    # ``DOF`` should not be allowed to vote for ``SDF`` or vice versa.
    def _is_ambiguous(t: str) -> bool:
        return len({e.domain for e in index.get(t, [])}) > 1

    context = {t for t in query_token_set if t != abbrev and t in index and not _is_ambiguous(t)}

    scores: dict[str, int] = {}
    for e in eligible:
        in_domain = domain_index.get(e.domain, set())
        scores[e.domain] = len(context & in_domain)

    if not scores:
        return None
    best_domain, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return None
    # Tie? Bail out.
    if sum(1 for s in scores.values() if s == best_score) > 1:
        return None
    for e in eligible:
        if e.domain == best_domain:
            return e
    return None


def expand(
    query: str,
    *,
    glossary: dict[str, list[str]] | None = None,
    domains: set[str] | None = None,
) -> str:
    """Return the query with known-abbreviation expansions appended.

    Parameters
    ----------
    query
        The user's raw search query.
    glossary
        A flat ``{ABBREV: [terms]}`` map, intended for tests. When
        provided, the domain-aware code path is bypassed entirely and
        every entry is treated as belonging to a single ``_external``
        domain (so ambiguity logic doesn't fire).
    domains
        Restrict expansion to abbreviations defined in the listed
        glossary categories. When ``None``, every category is eligible.
        Used by callers that know their search corpus is constrained
        (e.g. ``isaacsim_mcp`` passes ``{"physics", "sensors"}`` so that
        ``DOF`` expands to "Degree Of Freedom" rather than "Depth of
        Field"). Read from the ``OVAI_QE_DOMAINS`` env var if not
        passed (comma-separated; empty/unset means no scoping).

    Behaviour
    ---------
    Expansion is **additive**: the original tokens are preserved.
    Redundant terms already present in the query are skipped (REQ-QE-2).
    Genuinely ambiguous abbreviations (multiple eligible categories,
    no clear contextual winner) are *not* expanded — the original
    glossary's flat-merge would have appended every meaning, polluting
    every retriever leg.

    Example
    -------
    ::

        expand("how to fix SSS with PBR materials")
        # → "how to fix SSS with PBR materials (Subsurface Scattering) (Physically Based Rendering)"

        expand("rigid body DOF constraints")
        # → "rigid body DOF constraints (Degree Of Freedom)"
        #    (DOF disambiguates to physics because "rigid"/"body" share that domain)
    """
    if not query:
        return query

    if domains is None:
        env_domains = os.environ.get("OVAI_QE_DOMAINS", "").strip()
        if env_domains:
            domains = {d.strip() for d in env_domains.split(",") if d.strip()}

    if glossary is not None:
        # Test-style override: wrap as one-domain entries so the
        # rest of the pipeline doesn't need a separate code path.
        index: dict[str, list[_Entry]] = {str(k).upper(): [_Entry("_external", tuple(v))] for k, v in glossary.items()}
    else:
        index = _load_glossary_index()
    if not index:
        return query

    raw_tokens = _TOKEN_RE.findall(query)
    query_lower = query.lower()

    # Expand dotted identifiers into their subtokens so ``Gf.Matrix4d``
    # yields lookup candidates [``Gf.Matrix4d``, ``Gf``, ``Matrix4d``].
    seen: set[str] = set()
    tokens: list[str] = []
    for t in raw_tokens:
        for candidate in (t, *t.split(".")):
            if candidate and candidate not in seen:
                seen.add(candidate)
                tokens.append(candidate)
    upper_tokens = {t.upper() for t in tokens}

    appended: list[str] = []
    fired_abbrevs: list[str] = []

    for token in tokens:
        key = token.upper()
        entries = index.get(key)
        if not entries:
            continue

        eligible = _filter_eligible(entries, domains)
        if not eligible:
            continue

        chosen = _disambiguate(key, eligible, upper_tokens)
        if chosen is None:
            continue  # ambiguous; stay silent

        for term in chosen.terms:
            if term.lower() in query_lower:
                continue
            if any(term.lower() == a.lower() for a in appended):
                continue
            appended.append(term)
        fired_abbrevs.append(token)

    if not appended:
        return query

    if _log_enabled():
        logger.info(
            "query_expansion fired: abbrevs=%s  appended=%s  domains=%s",
            fired_abbrevs,
            appended,
            sorted(domains) if domains else "all",
        )

    return f"{query} ({', '.join(appended)})"


# Backwards-compat alias for the older importable name.
expand_query = expand


# ---------------------------------------------------------------------------
# Backwards-compat: legacy callers that read the flat dict directly.
# ---------------------------------------------------------------------------


def _load_default_glossary() -> dict[str, list[str]]:
    """Flat ``{ABBREV: [terms]}`` view of the merged glossary.

    Retained because earlier code (and the test suite) imports it
    directly. Returns the union of all categories' terms per abbrev,
    matching the pre-domain-aware behaviour.
    """
    flat: dict[str, list[str]] = {}
    for key, entries in _load_glossary_index().items():
        merged: list[str] = []
        for e in entries:
            for t in e.terms:
                if t not in merged:
                    merged.append(t)
        flat[key] = merged
    return flat
