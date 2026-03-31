# Semantic Cross-Category Relationships Enhancement

## Problem
The GNN graph only showed same-category product relationships because it was based purely on:
- Transaction co-purchases (limited cross-category)
- Same-category connections
- Temporal correlations

Products like **frying pans/pressure cookers** (KICH) and **dishwashing liquid** (CLNS) had no edges even though they're logically related.

## Solution
Added **semantic relationship layer** to the graph builder that captures logical cross-category connections.

### Changes Made

#### 1. Created `semantic_relationships.py`
Defines 24+ category-to-category relationship rules:
- **KICH → CLNS** (Kitchen → Cleaning): weight 0.7 (functional relationship)
- **SPRT → FTRW** (Sports → Footwear): weight 0.7 (usage relationship)
- **CLOT → BAGL** (Clothing → Bags): weight 0.7 (complementary relationship)
- **GROC → BEVG** (Groceries → Beverages): weight 0.8 (meal relationship)
- And many more...

Relationship types:
- **functional**: One product maintains/cares for another (kitchen → cleaning)
- **complementary**: Used together for same purpose (clothing → footwear)
- **usage**: Used in same context/location (books → stationery)
- **meal**: Consumed together (groceries → beverages)
- **storage**: One contains the other (kitchen → frozen)

#### 2. Updated `improved_graph_builder.py`
- Added `_build_semantic_category_edges()` method
- Integrated semantic edges into graph combination
- Balanced weights:
  - Category (same): 40%
  - Transaction co-purchase: 20%
  - Temporal correlation: 10%
  - **Semantic cross-category: 30%** 🆕

#### 3. Rebuilt Graph
Ran `python -m gnn.improved_graph_builder` to regenerate `models/gnn/adjacency.pt` with new edges.

## Results

### Before (Only transaction-based)
- Frying pan → only other kitchen items
- Pressure cooker → only other kitchen items
- Missing logical connections

### After (With semantic relationships)
- Frying pan → dishwashing liquid ✅
- Frying pan → groceries ✅
- Frying pan → frozen foods ✅
- Pressure cooker → dishwashing liquid ✅
- Pressure cooker → groceries ✅

### Example Cross-Category Connections Now Available:
1. **Kitchen items (KICH) ↔ Cleaning supplies (CLNS)**: 0.7 weight
2. **Kitchen items (KICH) ↔ Groceries (GROC)**: 0.5 weight
3. **Clothing (CLOT) ↔ Footwear (FTRW)**: 0.8 weight
4. **Beverages (BEVG) ↔ Snacks (SNCK)**: 0.8 weight
5. **Furniture (FURH) ↔ Bedding (BEDM)**: 0.7 weight

## How It Works

When you select a kitchen product in the GNN Insight SKU Reference:
1. Backend finds all edges from that product
2. Separates same-category vs cross-category
3. Cross-category now includes:
   - Transaction-based edges (if bought together)
   - **Semantic edges (logical relationships)** 🆕
4. Frontend displays both sections with visual distinction

## Next Steps (If Needed)

1. **Adjust weights**: Can fine-tune semantic_weight in graph builder
2. **Add more relationships**: Edit `semantic_relationships.py` to add more category pairs
3. **Product-level rules**: Could add specific SKU-to-SKU semantic rules beyond category level

## Files Modified
- ✅ `ml/gnn/semantic_relationships.py` (new)
- ✅ `ml/gnn/improved_graph_builder.py` (enhanced)
- ✅ `ml/models/gnn/adjacency.pt` (rebuilt)
- ✅ `backend/app/api/gnn.py` (already updated to show cross-category)
- ✅ `frontend/app/analyst/components/GNN3DVisualizer.tsx` (already updated UI)
