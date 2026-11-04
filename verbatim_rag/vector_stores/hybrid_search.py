"""
Hybrid search utilities for combining results from multiple search methods.
"""

import json
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .base import SearchResult

logger = logging.getLogger(__name__)


def sanitize_hybrid_weights(hybrid_weights: Dict[str, float]) -> Dict[str, float]:
    """
    Validate and filter hybrid weight config.

    Args:
        hybrid_weights: Dict mapping method names to weights

    Returns:
        Cleaned dict with only valid methods and positive weights

    Raises:
        ValueError: If no valid weights remain after validation
    """
    if not hybrid_weights:
        raise ValueError("hybrid_weights must be a non-empty dict")

    allowed_methods = {"dense", "sparse", "full_text"}
    cleaned: Dict[str, float] = {}

    for method, weight in hybrid_weights.items():
        if method not in allowed_methods:
            logger.warning("Ignoring unsupported hybrid method '%s'", method)
            continue
        if not isinstance(weight, (int, float)) or weight <= 0:
            logger.warning("Ignoring non-positive weight for method '%s': %s", method, weight)
            continue
        cleaned[method] = float(weight)

    if not cleaned:
        raise ValueError("No valid hybrid_weights after validation")

    return cleaned


def normalize_weights(
    results_by_method: Dict[str, List], weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Normalize weights for available search methods.

    Args:
        results_by_method: Dict mapping method names to result lists
        weights: Dict of weights for each method

    Returns:
        Normalized weights that sum to 1.0
    """
    available_weights = {m: weights.get(m, 0.0) for m in results_by_method}
    total_weight = sum(available_weights.values())
    if total_weight == 0:
        logger.warning(
            "No non-zero weights for available methods; using equal weights "
            f"for: {list(results_by_method.keys())}"
        )
        return {k: 1.0 / len(results_by_method) for k in results_by_method}
    return {k: v / total_weight for k, v in available_weights.items()}


def merge_hybrid_results(
    results_by_method: Dict[str, List],
    top_k: int,
    weights: Dict[str, float],
    rrf_k: int = 60,
    log_label: str = "",
) -> List[Dict]:
    """
    Merge search results from multiple methods using weighted RRF.

    Args:
        results_by_method: Dict mapping method names to result lists
        top_k: Number of results to return
        weights: Dict of weights for each method
        rrf_k: RRF constant (default 60)
        log_label: Label for logging

    Returns:
        Merged list of result dicts
    """
    normalized_weights = normalize_weights(results_by_method, weights)

    if log_label:
        logger.debug(
            "Hybrid merge (%s): methods=%s normalized_weights=%s rrf_k=%s top_k=%s",
            log_label,
            list(results_by_method.keys()),
            normalized_weights,
            rrf_k,
            top_k,
        )

    scores_by_id = {}
    hit_map = {}

    for method_name, results in results_by_method.items():
        weight = normalized_weights.get(method_name, 0.0)
        for rank, hit in enumerate(results):
            hit_id = hit.get("id")
            if not hit_id:
                continue
            rrf_score = 1.0 / (rrf_k + rank + 1)
            weighted_score = weight * rrf_score

            if hit_id not in scores_by_id:
                scores_by_id[hit_id] = 0.0
                hit_map[hit_id] = hit
            scores_by_id[hit_id] += weighted_score

    sorted_ids = sorted(scores_by_id.keys(), key=lambda id: scores_by_id[id], reverse=True)
    merged_results = []
    for hit_id in sorted_ids[:top_k]:
        hit = hit_map[hit_id].copy()
        hit["distance"] = 1.0 - scores_by_id[hit_id]
        merged_results.append(hit)

    return merged_results


def convert_hits_to_results(
    hits: List,
    dynamic_fields: Optional[List[str]] = None,
) -> List["SearchResult"]:
    """
    Convert raw hits to SearchResult objects.

    Args:
        hits: List of raw hit dicts from Milvus
        dynamic_fields: Optional list of dynamic field names to include in metadata

    Returns:
        List of SearchResult objects
    """
    from .base import SearchResult

    if dynamic_fields is None:
        dynamic_fields = []

    search_results: List[SearchResult] = []
    for hit in hits:
        entity = hit.get("entity", {})
        metadata = entity.get("metadata", {}) or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {"raw": metadata}

        for f in dynamic_fields:
            val = entity.get(f)
            if val is not None:
                metadata[f] = val

        search_results.append(
            SearchResult(
                id=hit.get("id"),
                score=hit.get("distance", 0.0),
                text=entity.get("text", ""),
                enhanced_text=entity.get("enhanced_text", ""),
                metadata=metadata,
            )
        )
    return search_results
