# Category Relationship System - Documentation

## Overview

This document explains the intelligent category-based demand propagation system that prevents spurious correlations in the GNN-based demand forecasting.

## The Problem (Before Fix)

### What Was Wrong?

The original GNN propagation blindly followed graph edges without considering whether product relationships made logical sense:

**Example**: "Rice prices increase"
- ❌ **WRONG Output**: Milk, Butter, Cheese demand +50%
- ❌ **Why Wrong**: Rice (GROC) has no logical relationship to dairy products (BKDY)
- ❌ **Root Cause**: GNN graph connected Rice ↔ Dairy through:
  - Co-purchase patterns (people buy both in same shopping trip)
  - Temporal correlation (both spike on weekends)
  - But **correlation ≠ causation**!

### Graph Edge Sources
The GNN graph creates edges using 3 signals:

1. **Category Edges (30% weight)**: ✅ Logical - products in same category
2. **Co-purchase Edges (50% weight)**: ⚠️ Noisy - includes spurious correlations
3. **Temporal Correlation Edges (20% weight)**: ⚠️ Noisy - correlation ≠ causation

## The Solution (After Fix)

### Category-Aware Propagation

Now, demand propagation **only occurs between logically related categories**:

```
Rice Price Increase (+50% demand)
  ↓
GROC → GROC: ✅ Full impact (same category)
GROC → BKDY: ✅ Moderate impact (+28%) - complements (rice + dairy meals)
GROC → FRPR: ✅ Moderate impact (+28%) - complements (rice + vegetables)
GROC → MEAT: ✅ Moderate impact (+28%) - complements (rice + meat)
GROC → BEVG: ✅ Moderate impact (+28%) - complements (meals + drinks)
GROC → CLOT: ❌ No impact - unrelated categories
GROC → ELEC: ❌ No impact - unrelated categories
```

## Category Relationships

### Food & Beverage Categories (High Propagation Strength: 0.8)

**GROC (Grocery - Staples & Grains)**
- Products: Rice, Cooking Oil, Wheat Flour, Sugar, Salt, Lentils, Pasta, Cereal, Tea, Coffee
- Complements: BKDY, FRPR, MEAT, BEVG
- Substitutes: BKDY, FRZN
- Logic: Grains are used with vegetables, dairy, meat in cooking

**BKDY (Bakery & Dairy)**
- Products: Bread, Butter, Cheese, Milk, Yogurt, Cream, Croissants
- Complements: GROC, BEVG, SNCK
- Substitutes: GROC, BKDY
- Logic: Dairy products complement bread, beverages, snacks

**FRPR (Fresh Produce)**
- Products: Apples, Bananas, Tomatoes, Potatoes, Onions, Carrots, Spinach
- Complements: GROC, BKDY, MEAT
- Substitutes: FRPR, FRZN
- Logic: Vegetables used with grains, dairy, meat

**MEAT (Meat & Seafood)**
- Products: Chicken, Fish, Prawns, Mutton, Salmon, Beef, Bacon
- Complements: GROC, FRPR, FRZN
- Substitutes: MEAT, FRZN
- Logic: Protein sources used with grains, vegetables

**BEVG (Beverages)**
- Products: Water, Cola, Juice, Beer, Wine, Energy Drinks
- Complements: GROC, BKDY, SNCK, FRZN
- Logic: Drinks complement meals and snacks

**SNCK (Snacks)**
- Products: Chips, Chocolate, Biscuits, Candy, Cookies
- Complements: BEVG, BKDY
- Logic: Snacks consumed with beverages

**FRZN (Frozen Foods)**
- Products: Frozen Peas, Fries, Nuggets, Pizza, Ice Cream
- Complements: GROC, BEVG, MEAT
- Logic: Frozen items as meal components or sides

### Household Categories (Medium Propagation Strength: 0.5)

**PRSN (Personal Care)**
- Products: Shampoo, Soap, Toothpaste, Deodorant, Lotion
- Complements: CLNS, BABC

**CLNS (Cleaning Supplies)**
- Products: Floor Cleaner, Detergent, Dishwashing Liquid, Mop
- Complements: PRSN, KICH

**BABC (Baby Care)**
- Products: Diapers, Wipes, Baby Food, Formula, Baby Shampoo
- Complements: PRSN, CLNS, BKDY

**KICH (Kitchen Appliances)**
- Products: Frying Pan, Pressure Cooker, Microwave, Blender
- Complements: GROC, CLNS

### Fashion Categories (Low Propagation Strength: 0.3)

**CLOT (Clothing)**
- Products: T-Shirts, Jeans, Jackets, Formal Wear
- Complements: FTRW, BAGL, JWCH

**FTRW (Footwear)**
- Products: Sneakers, Sandals, Formal Shoes, Boots
- Complements: CLOT, BAGL

**BAGL (Bags & Luggage)**
- Products: Backpack, Handbag, Suitcase, Laptop Bag
- Complements: CLOT, FTRW

**JWCH (Jewelry & Watches)**
- Products: Necklace, Ring, Earrings, Watches
- Complements: CLOT

### Home & Lifestyle Categories (Low Propagation Strength: 0.3)

**FURH (Furniture)**
- Products: Dining Table, Sofa, Bed, Wardrobe
- Complements: BEDM, CLNS, KICH

**BEDM (Bedding & Mattress)**
- Products: Mattress, Bedsheet, Pillow, Comforter
- Complements: FURH, CLNS

**ELEC (Electronics)**
- Products: Smartphone, TV, Laptop, Camera
- Complements: STOF, BAGL

**BOOK (Books & Media)**
- Products: Novels, Magazines, Comics, Textbooks
- Complements: STOF

**TOYG (Toys & Games)**
- Products: Teddy Bear, RC Car, Lego, Puzzles
- Complements: BOOK, BABC

**SPRT (Sports Equipment)**
- Products: Football, Cricket Bat, Yoga Mat, Bicycle
- Complements: CLOT, FTRW, BEVG

**STOF (Stationery & Office)**
- Products: Notebook, Pen, Pencil, Stapler
- Complements: None (independent)

### Specialized Categories (Very Low Propagation: 0.1)

**PETC (Pet Care)** - Independent category
**AUTO (Automotive)** - Independent category

## Propagation Math

### Formula

```python
propagation_multiplier = base_multiplier × category_strength × relationship_factor

Where:
- base_multiplier = original demand change (e.g., 1.5 for +50%)
- category_strength = 0.1 to 0.8 (based on category type)
- relationship_factor:
    - Complements: 0.7 (70% of impact propagates)
    - Substitutes: 0.4 (40% of impact propagates)
    - Unrelated: 0.0 (no propagation)
```

### Example Calculations

**Scenario**: Rice demand increases by 50% (multiplier = 1.5)

```
Source: GROC (strength = 0.8)

GROC → GROC (same):
  = 1.5 (full impact)

GROC → BKDY (complement):
  = 1.0 + (1.5 - 1.0) × 0.8 × 0.7
  = 1.0 + 0.5 × 0.8 × 0.7
  = 1.0 + 0.28
  = 1.28 (+28% impact)

GROC → FRPR (complement):
  = 1.28 (+28% impact)

GROC → CLOT (unrelated):
  = 1.0 (no impact)
```

## Benefits of This System

### ✅ Accuracy
- Eliminates spurious correlations
- Only propagates through logical relationships
- Matches real-world shopping behavior

### ✅ Interpretability
- Clear reasoning for why categories affect each other
- Easy to explain to business users
- Transparent decision-making

### ✅ Maintainability
- All rules in one config file
- Easy to adjust relationships
- Can add new categories easily

### ✅ Performance
- Filters out unnecessary propagations
- Reduces computation on unrelated products
- Faster impact calculations

## Usage

### For Developers

```python
from category_relationships import (
    are_categories_related,
    get_propagation_multiplier,
    CATEGORY_NAMES
)

# Check if categories are related
if are_categories_related('GROC', 'BKDY'):
    # Propagate demand

# Calculate propagated multiplier
multiplier = get_propagation_multiplier(
    source_cat='GROC',
    target_cat='BKDY',
    base_multiplier=1.5  # +50% demand
)
# Result: 1.28 (+28% propagated impact)
```

### For Business Users

When you ask "What happens if rice prices increase?", the system now:

1. ✅ **Correctly identifies** rice category (GROC)
2. ✅ **Only affects related categories**:
   - Other groceries (pasta, flour, etc.)
   - Vegetables (used in rice dishes)
   - Meat (used in rice dishes)
   - Beverages (consumed with meals)
3. ❌ **Does NOT affect unrelated categories**:
   - Clothing, Electronics, Furniture, etc.

## Testing

Run the test script to see all relationships:

```bash
cd ml
python config/category_relationships.py
```

Output shows:
- All category relationships
- Propagation strengths
- Example impact calculations

## Future Enhancements

1. **Seasonal Adjustments**: Higher propagation during holidays
2. **Store-Specific Rules**: Different patterns for different store types
3. **Dynamic Learning**: Adjust relationships based on actual purchase data
4. **Price Elasticity**: Factor in how price changes affect demand differently

## See Also

- `ml/config/category_relationships.py` - Configuration file
- `backend/app/services/gnn_propagation.py` - Implementation
- `ml/gnn/improved_graph_builder.py` - Graph construction
