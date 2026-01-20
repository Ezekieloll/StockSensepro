class InventoryRebalancer:
    """
    Greedy inter-store inventory rebalancer.
    Redistributes surplus to deficit stores for the SAME SKU.
    """

    def rebalance(self, sku, store_inventory, worst_case_demand):
        """
        Parameters
        ----------
        sku : str
            SKU identifier (for logging / traceability)
        store_inventory : dict
            {store_id: inventory_units}
        worst_case_demand : dict
            {store_id: worst_case_demand_units}

        Returns
        -------
        List of transfer actions:
        [
          {"sku": SKU, "from": store_A, "to": store_B, "units": X},
          ...
        ]
        """

        # -----------------------------
        # Step 1: classify stores
        # -----------------------------
        surplus = {}
        deficit = {}

        for store, inv in store_inventory.items():
            demand = worst_case_demand.get(store, 0)

            if inv > demand:
                surplus[store] = inv - demand
            elif inv < demand:
                deficit[store] = demand - inv

        # No action needed
        if not surplus or not deficit:
            return []

        # -----------------------------
        # Step 2: sort stores
        # -----------------------------
        surplus_sorted = sorted(
            surplus.items(), key=lambda x: x[1], reverse=True
        )
        deficit_sorted = sorted(
            deficit.items(), key=lambda x: x[1], reverse=True
        )

        # -----------------------------
        # Step 3: greedy transfers
        # -----------------------------
        transfers = []

        s_idx = 0
        for def_store, def_units in deficit_sorted:
            remaining_deficit = def_units

            while remaining_deficit > 0 and s_idx < len(surplus_sorted):
                sup_store, sup_units = surplus_sorted[s_idx]

                transfer_units = min(remaining_deficit, sup_units)

                if transfer_units > 0:
                    transfers.append({
                        "sku": sku,
                        "from": sup_store,
                        "to": def_store,
                        "units": transfer_units
                    })

                    # Update balances
                    remaining_deficit -= transfer_units
                    surplus_sorted[s_idx] = (
                        sup_store, sup_units - transfer_units
                    )

                # Move to next surplus store if exhausted
                if surplus_sorted[s_idx][1] == 0:
                    s_idx += 1

        return transfers
