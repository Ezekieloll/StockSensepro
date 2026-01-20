from adversarial.inventory_risk import InventoryRiskEvaluator

def main():
    risk = InventoryRiskEvaluator()

    result = risk.evaluate(
        baseline_demand=20,
        worst_case_demand=35,
        inventory_level=25,
    )

    print(result)

if __name__ == "__main__":
    main()
