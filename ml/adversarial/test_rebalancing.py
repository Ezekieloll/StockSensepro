from adversarial.rebalancing import InventoryRebalancer

def main():
    rebalancer = InventoryRebalancer()

    sku = "SKU_TEST"

    store_inventory = {
        "S1": 100,
        "S2": 20,
        "S3": 30
    }

    worst_case_demand = {
        "S1": 60,
        "S2": 50,
        "S3": 40
    }

    plan = rebalancer.rebalance(
        sku=sku,
        store_inventory=store_inventory,
        worst_case_demand=worst_case_demand
    )

    for t in plan:
        print(t)

if __name__ == "__main__":
    main()
