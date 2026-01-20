import torch


class InventoryRiskEvaluator:
    """
    Computes inventory risk metrics based on adversarial demand forecasts.
    Deterministic and interpretable.
    """

    def __init__(self, eps=1e-6):
        self.eps = eps

    def evaluate(
        self,
        baseline_demand,
        worst_case_demand,
        inventory_level,
    ):
        """
        Parameters
        ----------
        baseline_demand : Tensor or float
            Baseline forecasted daily demand
        worst_case_demand : Tensor or float
            Worst-case adversarial demand
        inventory_level : Tensor or float
            Current on-hand inventory

        Returns
        -------
        dict with:
            stockout (bool)
            severity (float)
            days_of_cover (float)
            risk_score (float)
        """

        # Convert to tensors for uniform handling
        bd = torch.as_tensor(baseline_demand, dtype=torch.float32)
        wd = torch.as_tensor(worst_case_demand, dtype=torch.float32)
        inv = torch.as_tensor(inventory_level, dtype=torch.float32)

        # Stockout condition
        stockout = wd > inv

        # Severity (units short)
        severity = torch.clamp(wd - inv, min=0.0)

        # Days of cover
        days_of_cover = inv / (bd + self.eps)

        # Risk score (simple normalized metric)
        # 0 → safe, 1 → critical
        risk_score = torch.clamp(severity / (wd + self.eps), 0.0, 1.0)

        return {
            "stockout": stockout.item() if stockout.numel() == 1 else stockout,
            "severity": severity.item() if severity.numel() == 1 else severity,
            "days_of_cover": days_of_cover.item() if days_of_cover.numel() == 1 else days_of_cover,
            "risk_score": risk_score.item() if risk_score.numel() == 1 else risk_score,
        }
