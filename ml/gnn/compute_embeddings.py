"""
Product Similarity using NLP Embeddings

Uses sentence transformers to compute semantic similarity between products
based on their names/descriptions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")


class ProductEmbeddings:
    """Compute and manage product embeddings for similarity."""
    
    def __init__(self, categories_path: str):
        """
        Initialize with product catalog.
        
        Args:
            categories_path: Path to categories_products.csv
        """
        # Read CSV, skipping header row
        self.categories = pd.read_csv(
            categories_path, 
            names=['Category_Code', 'SKU_ID', 'Product_Name'],
            skiprows=1  # Skip header row
        )
        self.model = None
        self.embeddings = None
        self.similarity_matrix = None
        
    def load_model(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Load sentence transformer model.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2 - fast & good quality)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available")
        
        print(f"\n📦 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("   ✓ Model loaded")
        
    def compute_embeddings(self):
        """Compute embeddings for all products."""
        if self.model is None:
            self.load_model()
        
        print("\n🧠 Computing product embeddings...")
        
        # Category descriptions for context
        category_descriptions = {
            'KICH': 'kitchen cooking utensil',
            'CLNS': 'cleaning supply',
            'GROC': 'grocery food item',
            'BEVG': 'beverage drink',
            'FRPR': 'fresh produce fruit vegetable',
            'MEAT': 'meat protein food',
            'BKDY': 'bakery bread pastry',
            'FRZN': 'frozen food',
            'SNCK': 'snack food',
            'CLOT': 'clothing apparel',
            'FTRW': 'footwear shoes',
            'BAGL': 'bag luggage',
            'JWCH': 'jewelry watch accessory',
            'PRSN': 'personal care hygiene',
            'BABC': 'baby care product',
            'TOYG': 'toy game',
            'ELEC': 'electronics device',
            'STOF': 'stationery office supply',
            'FURH': 'furniture home',
            'BEDM': 'bedding mattress',
            'PETC': 'pet care product',
            'SPRT': 'sports outdoor equipment',
            'AUTO': 'automotive car accessory',
            'BOOK': 'book media',
        }
        
        # Create rich descriptions for better embeddings
        descriptions = []
        for _, row in self.categories.iterrows():
            # Combine product name with category context
            cat_desc = category_descriptions.get(row['Category_Code'], '')
            desc = f"{cat_desc} {row['Product_Name']}"
            descriptions.append(desc)
        
        # Compute embeddings
        self.embeddings = self.model.encode(
            descriptions, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"   ✓ Computed {len(self.embeddings)} embeddings")
        print(f"   ✓ Embedding dimension: {self.embeddings.shape[1]}")
        
    def compute_similarity_matrix(self, min_similarity: float = 0.0):
        """
        Compute pairwise cosine similarity between all products.
        
        Args:
            min_similarity: Minimum similarity threshold (0-1)
        """
        if self.embeddings is None:
            self.compute_embeddings()
        
        print("\n📊 Computing similarity matrix...")
        
        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(self.embeddings)
        
        # Apply threshold
        self.similarity_matrix[self.similarity_matrix < min_similarity] = 0
        
        # Remove self-loops (will add back later in graph builder)
        np.fill_diagonal(self.similarity_matrix, 0)
        
        # Statistics
        non_zero = (self.similarity_matrix > 0).sum() / 2  # Symmetric matrix
        n_products = len(self.similarity_matrix)
        max_edges = n_products * (n_products - 1) / 2
        
        print(f"   ✓ Matrix shape: {self.similarity_matrix.shape}")
        print(f"   ✓ Non-zero edges: {int(non_zero)} / {int(max_edges)}")
        print(f"   ✓ Sparsity: {(1 - non_zero/max_edges)*100:.1f}%")
        print(f"   ✓ Min similarity: {self.similarity_matrix[self.similarity_matrix > 0].min():.3f}")
        print(f"   ✓ Max similarity: {self.similarity_matrix.max():.3f}")
        print(f"   ✓ Mean similarity: {self.similarity_matrix[self.similarity_matrix > 0].mean():.3f}")
        
    def save_results(self, output_dir: str):
        """
        Save embeddings and similarity matrix.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving results to {output_dir}...")
        
        # Save embeddings
        embeddings_file = output_path / "product_embeddings.npy"
        np.save(embeddings_file, self.embeddings)
        print(f"   ✓ Embeddings: {embeddings_file}")
        
        # Save similarity matrix
        similarity_file = output_path / "similarity_matrix.npy"
        np.save(similarity_file, self.similarity_matrix)
        print(f"   ✓ Similarity matrix: {similarity_file}")
        
        # Save SKU mapping for reference
        sku_mapping = self.categories[['SKU_ID', 'Product_Name', 'Category_Code']].to_dict('records')
        mapping_file = output_path / "sku_mapping.pkl"
        with open(mapping_file, 'wb') as f:
            pickle.dump(sku_mapping, f)
        print(f"   ✓ SKU mapping: {mapping_file}")
        
    def show_top_similar(self, sku: str, top_k: int = 10):
        """
        Show top similar products for a given SKU.
        
        Args:
            sku: SKU to query
            top_k: Number of similar products to show
        """
        if self.similarity_matrix is None:
            raise ValueError("Must compute similarity matrix first")
        
        # Find SKU index
        idx = self.categories[self.categories['SKU_ID'] == sku].index[0]
        
        # Get similarities for this SKU
        similarities = self.similarity_matrix[idx]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        print(f"\n🔍 Top {top_k} similar products to {sku}:")
        target_name = self.categories.iloc[idx]['Product_Name']
        print(f"   Target: {target_name}\n")
        
        for rank, other_idx in enumerate(top_indices, 1):
            other_sku = self.categories.iloc[other_idx]['SKU_ID']
            other_name = self.categories.iloc[other_idx]['Product_Name']
            other_cat = self.categories.iloc[other_idx]['Category_Code']
            similarity = similarities[other_idx]
            
            if similarity > 0:
                cross_cat = "🔗" if other_cat != self.categories.iloc[idx]['Category_Code'] else "  "
                print(f"   {rank:2d}. {cross_cat} {other_sku:15s} {other_name:30s} [{other_cat}] → {similarity:.3f}")


def main():
    """Main function to compute product embeddings."""
    
    # Paths
    categories_path = "data/raw/categories_products.csv"
    output_dir = "models/gnn"
    
    # Check if sentence-transformers is available
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n❌ sentence-transformers not installed!")
        print("   Please install: pip install sentence-transformers")
        return
    
    # Initialize
    embedder = ProductEmbeddings(categories_path)
    
    # Compute embeddings
    embedder.load_model('all-MiniLM-L6-v2')  # Fast, good quality model
    embedder.compute_embeddings()
    
    # Compute similarity (min threshold 0.35 to only keep strong relationships)
    embedder.compute_similarity_matrix(min_similarity=0.35)
    
    # Save results
    embedder.save_results(output_dir)
    
    # Show examples
    print("\n" + "="*60)
    print("📋 EXAMPLE SIMILARITIES")
    print("="*60)
    
    # Example: Frying Pan
    embedder.show_top_similar('SKU_KICH001', top_k=15)
    
    # Example: Pressure Cooker
    embedder.show_top_similar('SKU_KICH002', top_k=10)
    
    print("\n✅ Done! Use these embeddings in the graph builder.")
    print("   Next: Update improved_graph_builder.py to use similarity_matrix.npy")


if __name__ == "__main__":
    main()
