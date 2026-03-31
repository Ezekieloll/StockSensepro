"""
Semantic Category Relationships - MINIMAL & LOGICAL ONLY

Only includes relationships that make STRONG logical sense.
No weak or questionable connections.
"""

# Semantic relationship rules
# Format: {category: [(related_category, relationship_type, weight), ...]}
SEMANTIC_RELATIONSHIPS = {
    # Kitchen & Dining → Cleaning (ONLY dish-related)
    'KICH': [
        ('CLNS', 'functional', 0.3),  # Some cleaning products for dishes/kitchen
        ('GROC', 'usage', 0.4),        # Cooking ingredients
    ],
    
    # Cleaning Supplies → Kitchen (reverse)
    'CLNS': [
        ('KICH', 'functional', 0.3),   # Clean kitchen items
    ],
    
    # Clothing → Footwear, Bags (outfit)
    'CLOT': [
        ('FTRW', 'complementary', 0.6),
        ('BAGL', 'complementary', 0.5),
    ],
    
    # Footwear → Clothing
    'FTRW': [
        ('CLOT', 'complementary', 0.6),
    ],
    
    # Bags → Clothing
    'BAGL': [
        ('CLOT', 'complementary', 0.5),
    ],
    
    # Baby Care → Toys
    'BABC': [
        ('TOYG', 'usage', 0.6),
    ],
    
    # Toys → Baby Care
    'TOYG': [
        ('BABC', 'usage', 0.6),
    ],
    
    # Groceries → Beverages, Fresh Produce
    'GROC': [
        ('BEVG', 'meal', 0.7),
        ('FRPR', 'meal', 0.6),
        ('KICH', 'usage', 0.4),  # Cooking
    ],
    
    # Beverages → Groceries, Snacks
    'BEVG': [
        ('GROC', 'meal', 0.7),
        ('SNCK', 'meal', 0.6),
    ],
    
    # Fresh Produce → Groceries
    'FRPR': [
        ('GROC', 'meal', 0.6),
        ('KICH', 'usage', 0.4),  # Food prep
    ],
    
    # Snacks → Beverages
    'SNCK': [
        ('BEVG', 'meal', 0.6),
    ],
    
    # Meat → Fresh Produce, Groceries
    'MEAT': [
        ('FRPR', 'meal', 0.6),
        ('GROC', 'meal', 0.5),
        ('KICH', 'usage', 0.4),  # Cooking
    ],
    
    # Sports → Footwear, Clothing
    'SPRT': [
        ('FTRW', 'usage', 0.6),  # Sports shoes
        ('CLOT', 'usage', 0.5),  # Sports apparel
    ],
    
    # Bedding → Furniture (bedroom setup only)
    'BEDM': [
        ('FURH', 'usage', 0.5),
    ],
    
    # Furniture → Bedding (bedroom setup only)
    'FURH': [
        ('BEDM', 'usage', 0.5),
    ],
}


def get_semantic_edge_weight(category1: str, category2: str) -> float:
    """
    Get the semantic relationship weight between two categories.
    
    Args:
        category1: First category code
        category2: Second category code
        
    Returns:
        Weight (0-1) representing semantic relationship strength
    """
    # Check direct relationship
    if category1 in SEMANTIC_RELATIONSHIPS:
        for related_cat, rel_type, weight in SEMANTIC_RELATIONSHIPS[category1]:
            if related_cat == category2:
                return weight
    
    # Check reverse relationship
    if category2 in SEMANTIC_RELATIONSHIPS:
        for related_cat, rel_type, weight in SEMANTIC_RELATIONSHIPS[category2]:
            if related_cat == category1:
                return weight
    
    return 0.0


def get_all_semantic_edges():
    """
    Get all semantic relationships as a list of edges.
    
    Returns:
        List of (category1, category2, weight, relationship_type) tuples
    """
    edges = []
    processed = set()
    
    for cat1, relationships in SEMANTIC_RELATIONSHIPS.items():
        for cat2, rel_type, weight in relationships:
            # Avoid duplicates (bidirectional)
            edge_key = tuple(sorted([cat1, cat2]))
            if edge_key not in processed:
                processed.add(edge_key)
                edges.append((cat1, cat2, weight, rel_type))
    
    return edges


# Relationship type descriptions for documentation
RELATIONSHIP_TYPES = {
    'functional': 'One product is used to maintain/care for the other',
    'complementary': 'Products are used together for the same purpose',
    'usage': 'Products are used in the same context/location',
    'meal': 'Products are consumed together as part of a meal',
}
