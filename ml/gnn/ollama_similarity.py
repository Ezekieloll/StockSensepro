"""
Product Similarity using Local LLM (Ollama)

Uses a local LLM to intelligently rate product relationships
based on functional, complementary, and usage-based connections.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class OllamaProductSimilarity:
    """Compute product similarity using local LLM via Ollama."""
    
    def __init__(self, categories_path: str, ollama_url: str = "http://localhost:11434"):
        """
        Initialize with product catalog.
        
        Args:
            categories_path: Path to categories_products.csv
            ollama_url: Ollama API endpoint
        """
        # Read CSV, skipping header row
        self.categories = pd.read_csv(
            categories_path, 
            names=['Category_Code', 'SKU_ID', 'Product_Name'],
            skiprows=1
        )
        self.ollama_url = ollama_url
        self.similarity_matrix = None
        
        # Category descriptions for better context
        self.category_names = {
            'KICH': 'Kitchen & Dining',
            'CLNS': 'Cleaning Supplies',
            'GROC': 'Groceries',
            'BEVG': 'Beverages',
            'FRPR': 'Fresh Produce',
            'MEAT': 'Meat & Seafood',
            'BKDY': 'Bakery',
            'FRZN': 'Frozen Foods',
            'SNCK': 'Snacks',
            'CLOT': 'Clothing',
            'FTRW': 'Footwear',
            'BAGL': 'Bags & Luggage',
            'JWCH': 'Jewelry & Watches',
            'PRSN': 'Personal Care',
            'BABC': 'Baby Care',
            'TOYG': 'Toys & Games',
            'ELEC': 'Electronics',
            'STOF': 'Stationery & Office',
            'FURH': 'Furniture & Home',
            'BEDM': 'Bedding & Mattresses',
            'PETC': 'Pet Care',
            'SPRT': 'Sports & Outdoors',
            'AUTO': 'Automotive',
            'BOOK': 'Books & Media',
        }
        
    def query_ollama(self, prompt: str, model: str = "llama3.2") -> str:
        """
        Query Ollama API.
        
        Args:
            prompt: Prompt to send
            model: Model name (default: llama3.2)
            
        Returns:
            Response text
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent scoring
                        "num_predict": 10,   # We only need a short number response
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return "0.0"
    
    def get_similarity_score(self, product1: dict, product2: dict, model: str = "llama3.2") -> float:
        """
        Get LLM-based similarity score between two products.
        
        Args:
            product1: Dict with SKU_ID, Product_Name, Category_Code
            product2: Dict with SKU_ID, Product_Name, Category_Code
            model: Ollama model to use
            
        Returns:
            Similarity score (0-1)
        """
        cat1_name = self.category_names.get(product1['Category_Code'], product1['Category_Code'])
        cat2_name = self.category_names.get(product2['Category_Code'], product2['Category_Code'])
        
        prompt = f"""Rate the relationship between these two products on a scale of 0.0 to 1.0.

Product A: {product1['Product_Name']} (Category: {cat1_name})
Product B: {product2['Product_Name']} (Category: {cat2_name})

Consider:
- Functional relationship (one is used with/for the other)
- Complementary use (used together in same activity)
- Similar purpose or use case

Scoring guide:
- 1.0 = Very similar items (e.g., two types of pans)
- 0.8 = Strongly related (e.g., frying pan and cooking oil)
- 0.6 = Related/complementary (e.g., frying pan and dish soap)
- 0.4 = Weakly related (same general domain)
- 0.2 = Barely related
- 0.0 = Completely unrelated (e.g., frying pan and shoes)

Answer with ONLY the numeric score (e.g., 0.8), nothing else."""

        response = self.query_ollama(prompt, model)
        
        # Extract numeric score from response
        try:
            # Handle various response formats
            score_str = response.replace("Score:", "").replace("score:", "").strip()
            # Take first number found
            import re
            match = re.search(r'([0-9]*\.?[0-9]+)', score_str)
            if match:
                score = float(match.group(1))
                # Ensure in valid range
                score = max(0.0, min(1.0, score))
                return score
            else:
                print(f"Warning: Could not parse score from response: {response}")
                return 0.0
        except ValueError:
            print(f"Warning: Invalid score response: {response}")
            return 0.0
    
    def compute_similarity_matrix(
        self, 
        model: str = "llama3.2",
        same_category_score: float = 0.85,
        min_cross_category_score: float = 0.35,
        batch_size: int = 10,
        max_workers: int = 4,
        checkpoint_file: str = "models/gnn/similarity_checkpoint.pkl"
    ):
        """
        Compute similarity matrix using LLM.
        
        Args:
            model: Ollama model name
            same_category_score: Score for same-category products (auto-assigned)
            min_cross_category_score: Minimum score threshold for cross-category
            batch_size: Batch size for parallel processing
            max_workers: Number of parallel workers
            checkpoint_file: Path to checkpoint file for resume
        """
        n = len(self.categories)
        self.similarity_matrix = np.zeros((n, n))
        
        print(f"\n🤖 Computing LLM-based similarity using {model}...")
        print(f"   Products: {n}")
        
        # Load checkpoint if exists
        checkpoint_path = Path(checkpoint_file)
        processed_pairs = set()
        checkpoint_matrix = None
        
        if checkpoint_path.exists():
            print(f"\n📂 Found checkpoint file: {checkpoint_path}")
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    checkpoint_matrix = checkpoint_data['matrix']
                    processed_pairs = checkpoint_data['processed_pairs']
                    print(f"   ✓ Loaded checkpoint with {len(processed_pairs)} processed pairs")
                    print(f"   Resuming from where you left off...")
                    self.similarity_matrix = checkpoint_matrix
            except Exception as e:
                print(f"   ⚠️  Could not load checkpoint: {e}")
                print(f"   Starting fresh...")
        
        # First pass: Same category (auto-score, no LLM needed)
        if checkpoint_matrix is None:
            print(f"\n📦 Processing same-category pairs (score={same_category_score})...")
            same_cat_count = 0
            for i in range(n):
                for j in range(i+1, n):
                    if self.categories.iloc[i]['Category_Code'] == self.categories.iloc[j]['Category_Code']:
                        self.similarity_matrix[i, j] = same_category_score
                        self.similarity_matrix[j, i] = same_category_score
                        same_cat_count += 1
            
            print(f"   ✓ Auto-scored {same_cat_count} same-category pairs")
        else:
            same_cat_count = int((self.similarity_matrix > same_category_score - 0.01).sum() / 2)
            print(f"\n📦 Restored {same_cat_count} same-category pairs from checkpoint")
        
        # Second pass: Cross-category pairs (use LLM)
        all_cross_cat_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                if self.categories.iloc[i]['Category_Code'] != self.categories.iloc[j]['Category_Code']:
                    pair_key = f"{i},{j}"
                    if pair_key not in processed_pairs:
                        all_cross_cat_pairs.append((i, j))
        
        total_cross_cat = len(all_cross_cat_pairs) + len(processed_pairs)
        print(f"\n🧠 Cross-category pairs:")
        print(f"   Total: {total_cross_cat}")
        print(f"   Already processed: {len(processed_pairs)}")
        print(f"   Remaining: {len(all_cross_cat_pairs)}")
        
        if all_cross_cat_pairs:
            print(f"   Model: {model}")
            print(f"   Parallel workers: {max_workers}")
            print(f"   Estimated time: ~{len(all_cross_cat_pairs) * 2 / 60:.1f} minutes")
            print(f"\n   💡 You can press Ctrl+C to pause - progress will be saved!")
        
        def process_pair(pair_idx):
            i, j = all_cross_cat_pairs[pair_idx]
            prod1 = self.categories.iloc[i].to_dict()
            prod2 = self.categories.iloc[j].to_dict()
            score = self.get_similarity_score(prod1, prod2, model)
            return i, j, score
        
        # Process in parallel with progress bar and checkpointing
        processed = 0
        checkpoint_interval = 50  # Save checkpoint every 50 pairs
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_pair, idx): idx for idx in range(len(all_cross_cat_pairs))}
                
                with tqdm(total=len(all_cross_cat_pairs), desc="LLM queries", ncols=80) as pbar:
                    for future in as_completed(futures):
                        i, j, score = future.result()
                        
                        # Only store if above threshold
                        if score >= min_cross_category_score:
                            self.similarity_matrix[i, j] = score
                            self.similarity_matrix[j, i] = score
                        
                        # Mark as processed
                        processed_pairs.add(f"{i},{j}")
                        processed += 1
                        pbar.update(1)
                        
                        # Save checkpoint periodically
                        if processed % checkpoint_interval == 0:
                            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(checkpoint_path, 'wb') as f:
                                pickle.dump({
                                    'matrix': self.similarity_matrix,
                                    'processed_pairs': processed_pairs
                                }, f)
                            pbar.set_postfix_str(f"💾 Checkpoint saved ({processed}/{len(all_cross_cat_pairs)})")
                        
                        # Show preview
                        if processed % 20 == 0:
                            prod1_name = self.categories.iloc[i]['Product_Name']
                            prod2_name = self.categories.iloc[j]['Product_Name']
                            pbar.set_postfix_str(f"{prod1_name[:15]} ↔ {prod2_name[:15]} = {score:.2f}")
        
        except KeyboardInterrupt:
            print(f"\n\n⏸️  Paused! Saving checkpoint...")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'matrix': self.similarity_matrix,
                    'processed_pairs': processed_pairs
                }, f)
            print(f"   ✓ Progress saved to {checkpoint_path}")
            print(f"   ✓ Processed: {len(processed_pairs)}/{total_cross_cat}")
            print(f"\n   To resume: Run the same command again")
            print(f"   python -m gnn.ollama_similarity")
            return
        
        # Final checkpoint save
        if checkpoint_path.exists():
            checkpoint_path.unlink()  # Delete checkpoint on completion
            print(f"\n   ✓ Deleted checkpoint file (computation complete)")
        
        # Statistics
        non_zero = (self.similarity_matrix > 0).sum() / 2
        print(f"\n✅ Similarity matrix complete!")
        print(f"   Total edges: {int(non_zero)}")
        print(f"   Same-category: {same_cat_count}")
        print(f"   Cross-category (>{min_cross_category_score}): {int(non_zero - same_cat_count)}")
        print(f"   Score range: {self.similarity_matrix[self.similarity_matrix > 0].min():.3f} - {self.similarity_matrix.max():.3f}")
    
    def save_results(self, output_dir: str):
        """Save similarity matrix."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving results to {output_dir}...")
        
        # Save similarity matrix
        similarity_file = output_path / "similarity_matrix.npy"
        np.save(similarity_file, self.similarity_matrix)
        print(f"   ✓ Similarity matrix: {similarity_file}")
        
        # Save metadata
        metadata = {
            'method': 'ollama_llm',
            'products': len(self.categories),
            'edges': int((self.similarity_matrix > 0).sum() / 2),
            'score_range': [
                float(self.similarity_matrix[self.similarity_matrix > 0].min()),
                float(self.similarity_matrix.max())
            ]
        }
        metadata_file = output_path / "similarity_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Metadata: {metadata_file}")
    
    def show_top_similar(self, sku: str, top_k: int = 10):
        """Show top similar products for a given SKU."""
        if self.similarity_matrix is None:
            raise ValueError("Must compute similarity matrix first")
        
        idx = self.categories[self.categories['SKU_ID'] == sku].index[0]
        similarities = self.similarity_matrix[idx]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        print(f"\n🔍 Top {top_k} similar products to {sku}:")
        target_name = self.categories.iloc[idx]['Product_Name']
        target_cat = self.categories.iloc[idx]['Category_Code']
        print(f"   Target: {target_name} [{target_cat}]\n")
        
        for rank, other_idx in enumerate(top_indices, 1):
            other_sku = self.categories.iloc[other_idx]['SKU_ID']
            other_name = self.categories.iloc[other_idx]['Product_Name']
            other_cat = self.categories.iloc[other_idx]['Category_Code']
            similarity = similarities[other_idx]
            
            if similarity > 0:
                cross_cat = "🔗" if other_cat != target_cat else "  "
                print(f"   {rank:2d}. {cross_cat} {other_sku:15s} {other_name:30s} [{other_cat}] → {similarity:.3f}")


def main():
    """Main function to compute LLM-based product similarities."""
    
    # Paths
    categories_path = "data/raw/categories_products.csv"
    output_dir = "models/gnn"
    
    print("="*70)
    print("🚀 LLM-BASED PRODUCT SIMILARITY COMPUTATION")
    print("="*70)
    print("\n⚡ Using your RTX 4070 for fast local inference!")
    print("   Make sure Ollama is running: ollama serve")
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get('models', [])
        print(f"\n✅ Ollama is running!")
        print(f"   Available models: {[m['name'] for m in models]}")
        
        # Recommend best model
        preferred_models = ['llama3.2', 'llama3.1', 'llama3', 'mistral', 'phi3']
        model_to_use = None
        for preferred in preferred_models:
            if any(preferred in m['name'] for m in models):
                model_to_use = next(m['name'] for m in models if preferred in m['name'])
                break
        
        if not model_to_use and models:
            model_to_use = models[0]['name']
        
        if not model_to_use:
            print("\n❌ No models found! Please pull a model first:")
            print("   ollama pull llama3.2")
            return
        
        print(f"   Using model: {model_to_use}")
        
    except Exception as e:
        print(f"\n❌ Cannot connect to Ollama: {e}")
        print("   Please start Ollama first: ollama serve")
        print("   Or install it from: https://ollama.ai")
        return
    
    # Initialize
    similarity = OllamaProductSimilarity(categories_path)
    
    # Compute similarities
    similarity.compute_similarity_matrix(
        model=model_to_use,
        same_category_score=0.85,          # Same category = strong relationship
        min_cross_category_score=0.4,      # Only keep cross-category if score >= 0.4
        max_workers=4                      # Parallel queries (your CPU can handle it!)
    )
    
    # Save results
    similarity.save_results(output_dir)
    
    # Show examples
    print("\n" + "="*70)
    print("📋 EXAMPLE SIMILARITIES")
    print("="*70)
    
    # Example: Frying Pan
    similarity.show_top_similar('SKU_KICH001', top_k=15)
    
    print("\n✅ Done! LLM-based similarity matrix is ready!")
    print("   Next step: python -m gnn.improved_graph_builder")


if __name__ == "__main__":
    main()
