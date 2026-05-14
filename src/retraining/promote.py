"""Champion vs. challenger comparison logic."""
import logging

logger = logging.getLogger(__name__)

# How much better must the challenger be to justify replacing the champion?
# Tiny gains aren't worth the deployment risk.
PROMOTION_MARGIN = 0.005  # 0.5 percentage points of AUC-PR


def should_promote(champion_metrics: dict, challenger_metrics: dict) -> tuple[bool, str]:
    """
    Returns (should_promote, reason).

    The challenger must beat the champion on AUC-PR by at least PROMOTION_MARGIN
    AND not regress on recall by more than 2 percentage points.
    """
    champ_auc_pr = champion_metrics.get("auc_pr", 0.0)
    chall_auc_pr = challenger_metrics.get("auc_pr", 0.0)
    champ_recall = champion_metrics.get("recall", 0.0)
    chall_recall = challenger_metrics.get("recall", 0.0)

    auc_pr_gain = chall_auc_pr - champ_auc_pr
    recall_drop = champ_recall - chall_recall

    if auc_pr_gain < PROMOTION_MARGIN:
        return False, (
            f"Challenger AUC-PR ({chall_auc_pr:.4f}) does not exceed champion "
            f"({champ_auc_pr:.4f}) by required margin ({PROMOTION_MARGIN})."
        )

    if recall_drop > 0.02:
        return False, (
            f"Challenger recall dropped by {recall_drop:.4f} — too risky to promote "
            f"(missed fraud has high business cost)."
        )

    return True, (
        f"Challenger wins: AUC-PR +{auc_pr_gain:.4f}, recall delta {-recall_drop:+.4f}."
    )