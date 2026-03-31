# Semantic Relationships Refinement

## Issue Identified
After implementing semantic relationships, we noticed **over-broad connections**:
- ❌ Frying Pan → Mop Set (doesn't make sense)
- ❌ Pressure Cooker → Toilet Cleaner (ridiculous)  
- ✅ Frying Pan → Dishwashing Liquid (makes sense!)

## Root Cause
Semantic relationships work at the **CATEGORY level**, not product level:
- `KICH (Kitchen)` → `CLNS (Cleaning Supplies)` with weight 0.7
- This connected **ALL** kitchen items to **ALL** cleaning products
- Created sensible links (cookware → dish soap) but also silly ones (cookware → mop)

## Solution Applied

### 1. Reduced Relationship Weights
Lowered overly broad category connections:
- `KICH → CLNS`: **0.7 → 0.4** (still exists, but less dominant)
- `CLNS → KICH`: **0.7 → 0.4** (matching)
- `CLNS → PRSN`: **0.5 → 0.4**
- `CLNS → CLOT`: **0.4 → 0.3**

### 2. Increased Edge Threshold
Raised the minimum weight threshold to keep an edge:
- Previous: **0.1**
- New: **0.15**

This filters out weak semantic-only connections.

### 3. Relying on Multi-Signal Combination
The final edge weight is a combination of:
```
Edge Weight = 
  0.4 × same_category +
  0.2 × co-purchase_data +
  0.1 × temporal_correlation +
  0.3 × semantic_relationship
```

**For Frying Pan → Dishwashing Liquid:**
- Same category: NO → 0
- Co-purchase: Moderate (if people buy together) → ~0.3  
- Temporal: Low → ~0.1
- Semantic: KICH→CLNS → 0.4 × 0.3 = 0.12

**Combined**: 0 + 0.06 + 0.01 + 0.12 = **0.19** ✅ (above threshold 0.15, KEEPS edge)

**For Frying Pan → Mop Set:**
- Same category: NO → 0
- Co-purchase: Very Low (rarely bought together) → ~0.05
- Temporal: Low → ~0.05
- Semantic: KICH→CLNS → 0.4 × 0.3 = 0.12

**Combined**: 0 + 0.01 + 0.005 + 0.12 = **0.135** ❌ (below threshold 0.15, REMOVES edge)

## Result
Now semantic relationships provide a **baseline hint** of possible connections, but:
- **Transaction data strengthens** sensible connections (dish soap, scrub pads)
- **Lack of transaction data filters out** silly connections (mop sets, toilet cleaner)
- Threshold eliminates edges that are purely semantic without data support

## What You Should See Now
When you select a frying pan or pressure cooker:
- ✅ Strong link to **Dishwashing Liquid** (semantic + transaction data)
- ✅ Moderate link to **Scrub Pads** (semantic + some transaction data)
- ✅ Links to **Groceries** (semantic: cooking ingredients)
- ❌ NO link to Mop Set (filtered by threshold)
- ❌ NO link to Toilet Cleaner (filtered by threshold)

## Philosophy
**Category-level semantic relationships** provide:
- Broad structural hints (kitchen relates to cleaning)
- Cross-category discovery opportunities
- Fill gaps where transaction data is sparse

**Product-level transaction data** provides:
- Specific validation (which cleaning products specifically?)
- Refinement of semantic hints
- Real-world shopping patterns

**Combined approach** = Best of both worlds! 🎯
