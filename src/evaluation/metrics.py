"""Evaluation metrics for comparing MetaHPO-C against standard HPO baselines."""

from typing import Any, List


def compute_simple_regret(best_observed: float, optimum_reference: float) -> float:
    """Compute simple regret between the best observed score and reference optimum.

    Parameters
    ----------
    best_observed : float
        Best validation score achieved by the optimization method up to trial t.
    optimum_reference : float
        Known or estimated optimal validation score on the dataset.

    Returns
    -------
    float
        Non-negative regret value (optimum_reference - best_observed).

    Raises
    ------
    NotImplementedError
        Metric calculation will be implemented in Phase 6.
    """
    # TODO: In Phase 6, compute standardized regret for maximization objectives.
    raise NotImplementedError(
        "Simple regret metric is not yet implemented. "
        "Benchmark evaluation metrics are scheduled for Phase 6."
    )


def compute_average_rank(trajectories: List[Any]) -> Any:
    """Compute average rank across datasets over optimization trials.

    Parameters
    ----------
    trajectories : list
        List of optimization trajectories for competing methods.

    Returns
    -------
    Any
        Rank progression over optimization budgets.

    Raises
    ------
    NotImplementedError
        Ranking metrics will be implemented in Phase 6.
    """
    # TODO: In Phase 6, rank methods per trial and aggregate average rank across test datasets.
    raise NotImplementedError(
        "Average rank calculation is not yet implemented. "
        "Ranking aggregations are scheduled for Phase 6."
    )
