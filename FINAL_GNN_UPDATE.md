# Final GNN Influences Update

## Changes Made (2026-02-10)

### 1. Simplified UI - Single Sorted List
**Removed** the separated "Same Category" and "Cross-Category" sections.
**Now shows** a single list sorted by connection strength (highest to lowest).

**Visual indicators:**
- **Rank number** (#1, #2, #3...) on the left
- **Larger ring + accent color** for cross-category items
- **Category badge** shown only for cross-category items
- **Subtle background color** difference (accent tint for cross-category)

### 2. Simplified Backend API
The `/gnn/product-influences/{sku}` endpoint now:
- Returns influences sorted purely by edge weight (descending)
- No artificial category separation
- Includes `is_cross_category` flag for frontend styling
- Top 20 results by default

### 3. Improved Graph Weights
**Final weight configuration:**
```python
category_weight    = 0.3  (30%)  # Same category baseline
copurchase_weight  = 0.5  (50%)  # ACTUAL shopping behavior ⭐
temporal_weight    = 0.0  ( 0%)  # Disabled (spurious correlations)
semantic_weight    = 0.2  (20%)  # Logical hints (not dominant)
```

**Philosophy:**
- **Transaction data is king** (50%) - what people ACTUALLY buy together
- **Category provides context** (30%) - same-category products are naturally related
- **Semantic adds intelligence** (20%) - fills gaps with logical relationships
- **Temporal disabled** (0%) - correlation ≠ causation

### 4. Edge Threshold
Raised from **0.1 → 0.15** to filter weak connections and prevent noise.

## Result

### What You See Now:
Influences sorted by **actual connection strength**, regardless of category:

```
#1  SKU_KICH002 (same) - Weight 0.95 🔵
#2  SKU_GROC005 (cross) - Weight 0.87 🟠 GROC
#3  SKU_KICH007 (same) - Weight 0.82 🔵
#4  SKU_CLNS003 (cross) - Weight 0.71 🟠 CLNS  ← Dishwashing liquid
#5  SKU_KICH001 (same) - Weight 0.68 🔵
...
```

### For Frying Pan / Pressure Cooker:
Top influences based on:
1. **Other kitchen items** (bought together, same category)
2. **Groceries** (cooking ingredients - strong co-purchase)
3. **Dishwashing liquid** (functional relationship + some co-purchase)
4. **Scrub pads** (cleaning cookware - functional relationship)

**NO MORE:**
- ❌ Mop sets (filtered out - weak connection)
- ❌ Toilet cleaner (filtered out - no real connection)
- ❌ Artificial category separation

## Files Modified
- ✅ `backend/app/api/gnn.py` - Simplified API
- ✅ `frontend/app/analyst/components/GNN3DVisualizer.tsx` - Single sorted list UI
- ✅ `ml/gnn/improved_graph_builder.py` - Better weights (50% co-purchase)
- ✅ `ml/gnn/semantic_relationships.py` - Reduced overly broad weights
- ✅ `ml/models/gnn/adjacency.pt` - Rebuilt graph with new weights

## Summary
The influences now show **what actually matters** - products with the strongest real connections, sorted by strength. Cross-category items are subtly highlighted but not artificially segregated.
