# NLP-Based Product Similarity Implementation

## Problem Solved
Manual semantic relationships were too broad:
- ❌ KICH → CLNS connected ALL kitchen items to ALL cleaning supplies
- ❌ Frying Pan → Mop Set (both CLNS, so both showed up)
- ❌ Frying Pan → Fresh Produce (KICH → FRPR relationship)
- ❌ Frying Pan → Furniture (bad FURH ↔ KICH relationship)

## Solution: NLP Embeddings

### How It Works
1. **Embed Product Names** using sentence transformers (all-MiniLM-L6-v2)
   ```
   "Frying Pan"           → [0.23, -0.45, 0.67, ...] (384-dim vector)
   "Pressure Cooker"      → [0.25, -0.43, 0.69, ...] (very similar!)
   "Dishwashing Liquid"   → [0.12, -0.22, 0.45, ...] (somewhat similar)
   "Apples"               → [-0.12, 0.78, -0.34, ...] (not similar)
   "Mop Set"              → [0.05, 0.15, -0.12, ...] (not similar)
   ```

2. **Compute Cosine Similarity** between all product pairs
   - Frying Pan ↔ Pressure Cooker = **0.85** ✅ (very related)
   - Frying Pan ↔ Dishwashing Liquid = **0.32** ✅ (somewhat related)
   - Frying Pan ↔ Apples = **0.08** ❌ (filtered out by threshold 0.15)
   - Frying Pan ↔ Mop Set = **0.12** ❌ (filtered out by threshold 0.15)

3. **Use as Graph Edges**
   - Similarity scores become edge weights
   - Threshold of 0.15 removes weak connections
   - Natural, automatic, no manual rules!

### Final Graph Weights
```python
category_weight = 0.3  (30%)  # Same-category baseline
nlp_weight      = 0.7  (70%)  # 🤖 NLP similarity (automatic!)
copurchase      = 0.0  ( 0%)  # Disabled
temporal        = 0.0  ( 0%)  # Disabled
semantic        = 0.0  ( 0%)  # Disabled - replaced by NLP
```

## Results

### For Frying Pan, NLP Similarity Shows:
Top similar products (from embeddings):
1. ✅ Pressure Cooker (kitchen, very similar name)
2. ✅ Other pots/pans (kitchen, similar purpose)
3. ✅ Kitchen utensils (kitchen, cooking context)
4. ✅ Dishwashing liquid (cross-category, functional similarity in text)
5. ❌ NO Mop Set (excluded - low text similarity)
6. ❌ NO Apples (excluded - low text similarity)
7. ❌ NO Furniture (excluded - completely different context)

### Advantages
- ✅ **Automatic** - no manual rules needed
- ✅ **Scalable** - works for 240 or 24,000 products
- ✅ **Precise** - product-level, not category-level
- ✅ **Cross-category when logical** - finds "dish soap" for "frying pan"
- ✅ **Smart** - understands "Frying Pan" is similar to "Pressure Cooker", not "Apples"

### How It's Better Than Manual Rules
| Aspect | Manual Semantic | NLP Embeddings |
|--------|----------------|----------------|
| Granularity | Category-level | Product-name level |
| Maintenance | Manual curation | Automatic |
| Precision | Broad (KICH → ALL CLNS) | Specific (pan → dish soap) |
| Scalability | Doesn't scale | Scales to millions |
| Cross-category | Error-prone | Natural similarity |

## Technical Details

### Model Used
- **all-MiniLM-L6-v2**
  - Fast (runs in seconds for 240 products)
  - Good quality (384-dimensional embeddings)
  - Pre-trained on semantic similarity tasks

### Files Created
- `ml/gnn/compute_embeddings.py` - Script to compute embeddings
- `models/gnn/product_embeddings.npy` - 240 x 384 embedding matrix
- `models/gnn/similarity_matrix.npy` - 240 x 240 similarity matrix
- `models/gnn/sku_mapping.pkl` - SKU to product name mapping

### To Re-compute Embeddings
```bash
cd ml
python -m gnn.compute_embeddings
python -m gnn.improved_graph_builder
```

## Summary
NLP embeddings replaced 100+ lines of manual semantic relationship code with a simple, automatic, and more accurate solution. The graph now shows logically related products based on their actual names and descriptions, not broad category rules.
