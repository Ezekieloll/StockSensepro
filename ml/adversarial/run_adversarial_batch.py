import torch
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from forecasting.dataset2 import DemandDatasetV2
from adversarial.scenario_simulator import ScenarioSimulator
from adversarial.inventory_risk import InventoryRiskEvaluator


# -----------------------------
# CONFIG
# -----------------------------

WINDOW_SIZE = 17
BATCH_SIZE = 256
STORM_WEATHER_CODE = 3.0   # adjust if encoding differs

MODEL_PATH = "models/best_model.pt"   # adjust if needed
DATA_PATH = "data/processed2/val.csv"


# -----------------------------
# LOAD INVENTORY (TEMP)
# -----------------------------
# Later this comes from DB
def load_inventory():
    """
    Returns:
        inventory[sku][store] = units
    """
    # MOCK inventory for now
    return defaultdict(lambda: {
        "S1": 50,
        "S2": 40,
        "S3": 30
    })


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = DemandDatasetV2(DATA_PATH, WINDOW_SIZE)

    # Load trained model
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.to(device)
    model.eval()

    simulator = ScenarioSimulator(model, device)
    risk_eval = InventoryRiskEvaluator()
    inventory = load_inventory()

    # Group latest sample per SKU per store
    latest_samples = {}
    for i in range(len(dataset)):
        x, _, sku = dataset[i]
        latest_samples[(sku, i)] = x

    results = []

    print("\n🚀 Running per-SKU adversarial evaluation...\n")

    for (sku, idx), x in tqdm(latest_samples.items()):
        x = x.unsqueeze(0)  # (1, W, F)

        # Adversarial forecasts
        baseline = simulator.baseline(x).item()
        worst = simulator.worst_case(x, weather_code=STORM_WEATHER_CODE).item()

        # Inventory per store
        for store_id, inv_units in inventory[sku].items():
            risk = risk_eval.evaluate(
                baseline_demand=baseline,
                worst_case_demand=worst,
                inventory_level=inv_units
            )

            results.append({
                "sku": sku,
                "store": store_id,
                "baseline_demand": baseline,
                "worst_case_demand": worst,
                "inventory": inv_units,
                **risk
            })

    df = pd.DataFrame(results)

    df.to_csv("analysis/adversarial_risk_per_sku.csv", index=False)
    print("\n✅ Saved: analysis/adversarial_risk_per_sku.csv")

    print("\n📊 SUMMARY")
    print(df[["stockout", "severity", "risk_score"]].describe())


if __name__ == "__main__":
    main()
