import torch


class ScenarioSimulator:
    """
    Adversarial scenario simulator for demand forecasting models.
    Inference-only. No training. No randomness.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.model.eval()
        self.device = device or next(model.parameters()).device

    # --------------------------------------------------
    # Internal helper
    # --------------------------------------------------

    def _predict(self, x):
        """
        Run model inference safely.
        """
        with torch.no_grad():
            return self.model(x.to(self.device)).cpu()

    def _clone(self, x):
        """
        Clone input tensor to avoid mutation.
        """
        return x.clone().detach()

    # --------------------------------------------------
    # 1️⃣ Baseline
    # --------------------------------------------------

    def baseline(self, x):
        """
        Baseline forecast without perturbation.
        """
        return self._predict(x)

    # --------------------------------------------------
    # 2️⃣ Demand Spike
    # --------------------------------------------------

    def demand_spike(self, x, factor=1.5):
        """
        Simulate sudden demand surge.
        """
        x_adv = self._clone(x)
        x_adv[:, :, 0] *= factor   # daily_demand
        return self._predict(x_adv)

    # --------------------------------------------------
    # 3️⃣ Demand Drop
    # --------------------------------------------------

    def demand_drop(self, x, factor=0.5):
        """
        Simulate sudden demand collapse.
        """
        x_adv = self._clone(x)
        x_adv[:, :, 0] *= factor
        return self._predict(x_adv)

    # --------------------------------------------------
    # 4️⃣ Weather Shock
    # --------------------------------------------------

    def weather_shock(self, x, target_weather_code):
        """
        Force extreme weather condition.
        target_weather_code: numeric encoding (e.g. Storm)
        """
        x_adv = self._clone(x)
        x_adv[:, :, 3] = target_weather_code
        return self._predict(x_adv)

    # --------------------------------------------------
    # 5️⃣ Holiday Shock
    # --------------------------------------------------

    def holiday_shock(self, x):
        """
        Force holiday flag ON.
        """
        x_adv = self._clone(x)
        x_adv[:, :, 2] = 1.0
        return self._predict(x_adv)

    # --------------------------------------------------
    # 6️⃣ Worst-case Envelope
    # --------------------------------------------------

    def worst_case(self, x, weather_code):
        """
        Compute worst-case demand across adversarial scenarios.
        """
        preds = torch.stack([
            self.baseline(x),
            self.demand_spike(x, 1.5),
            self.demand_spike(x, 2.0),
            self.weather_shock(x, weather_code),
            self.holiday_shock(x),
        ], dim=0)

        return preds.max(dim=0).values
