"""Paired Clean-versus-perturbed robustness reporting."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from robust_tool.data.perturb import PERTURBATION_KINDS
from robust_tool.data.schemas import Task
from robust_tool.eval.metrics import MetricValue, aggregate_metrics


def compute_robustness_report(
    clean_evaluations: Iterable[Mapping[str, Any]],
    robust_tasks: Iterable[Task],
    robust_evaluations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare every perturbed task with its frozen Clean source task."""

    clean_by_id = _index_evaluations(clean_evaluations, "Clean")
    robust_by_id = _index_evaluations(robust_evaluations, "Robust")
    tasks = list(robust_tasks)
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate Robust task IDs")
    if set(task_ids) != set(robust_by_id):
        missing = sorted(set(task_ids) - set(robust_by_id))
        extra = sorted(set(robust_by_id) - set(task_ids))
        raise ValueError(f"Robust task/evaluation mismatch; missing={missing}, extra={extra}")

    pairs_by_kind: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for task in tasks:
        robustness = task.metadata.get("robustness")
        if not isinstance(robustness, Mapping):
            raise ValueError(f"task lacks robustness metadata: {task.task_id}")
        kind = str(robustness.get("kind"))
        source_id = str(robustness.get("source_task_id"))
        if kind not in PERTURBATION_KINDS:
            raise ValueError(f"unknown perturbation kind for {task.task_id}: {kind}")
        if source_id not in clean_by_id:
            raise ValueError(f"Clean source evaluation is missing: {source_id}")
        pairs_by_kind[kind].append((clean_by_id[source_id], robust_by_id[task.task_id]))

    missing_kinds = set(PERTURBATION_KINDS) - set(pairs_by_kind)
    if missing_kinds:
        raise ValueError(f"Robust report is missing perturbations: {sorted(missing_kinds)}")
    settings = {
        kind: _summarize_pairs(pairs_by_kind[kind])
        for kind in PERTURBATION_KINDS
    }
    all_pairs = [pair for kind in PERTURBATION_KINDS for pair in pairs_by_kind[kind]]
    return {
        "pair_count": len(all_pairs),
        "overall": _summarize_pairs(all_pairs),
        "settings": settings,
    }


def _index_evaluations(
    evaluations: Iterable[Mapping[str, Any]],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for evaluation in evaluations:
        task_id = str(evaluation.get("task_id", ""))
        if not task_id:
            raise ValueError(f"{label} evaluation has no task_id")
        if task_id in indexed:
            raise ValueError(f"duplicate {label} evaluation task ID: {task_id}")
        diagnostics = evaluation.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError(f"{label} evaluation has no diagnostics: {task_id}")
        indexed[task_id] = evaluation
    return indexed


def _summarize_pairs(
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    clean = [pair[0] for pair in pairs]
    perturbed = [pair[1] for pair in pairs]
    clean_successes = sum(bool(item.get("success")) for item in clean)
    perturbed_successes = sum(bool(item.get("success")) for item in perturbed)
    count = len(pairs)
    clean_rate = MetricValue.rate(clean_successes, count)
    perturbed_rate = MetricValue.rate(perturbed_successes, count)
    repaired = sum(
        not bool(left.get("success")) and bool(right.get("success"))
        for left, right in pairs
    )
    regressed = sum(
        bool(left.get("success")) and not bool(right.get("success"))
        for left, right in pairs
    )
    failure_counts = Counter(
        str(failure)
        for item in perturbed
        for failure in item.get("failures", [])
    )
    return {
        "pair_count": count,
        "clean_task_success": clean_rate.to_dict(),
        "perturbed_task_success": perturbed_rate.to_dict(),
        "robustness_gap": (
            None
            if clean_rate.value is None or perturbed_rate.value is None
            else clean_rate.value - perturbed_rate.value
        ),
        "paired_transitions": {
            "clean_fail_perturbed_success": repaired,
            "clean_success_perturbed_fail": regressed,
        },
        "clean_metrics": _metrics_dict(clean),
        "perturbed_metrics": _metrics_dict(perturbed),
        "perturbed_failure_counts": dict(sorted(failure_counts.items())),
    }


def _metrics_dict(evaluations: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = aggregate_metrics(item["diagnostics"] for item in evaluations)
    return {name: metric.to_dict() for name, metric in metrics.items()}
