"""
Product Type Taxonomy for Fine-Grained Semantic Relationships

This goes beyond category codes to define product types,
enabling more precise logical connections.
"""

# Product type definitions
PRODUCT_TYPES = {
    # Kitchen items
    'SKU_KICH001': 'cookware',
    'SKU_KICH002': 'cookware',
    'SKU_KICH003': 'cookware',
    'SKU_KICH004': 'cookware',
    'SKU_KICH005': 'utensils',
    'SKU_KICH006': 'utensils',
    'SKU_KICH007': 'storage',
    'SKU_KICH008': 'storage',
    'SKU_KICH009': 'appliance',
    'SKU_KICH010': 'appliance',
    
    # Cleaning supplies
    'SKU_CLNS001': 'floor_cleaning',
    'SKU_CLNS002': 'bathroom_cleaning',
    'SKU_CLNS003': 'dish_cleaning',      # Dishwashing Liquid
    'SKU_CLNS004': 'surface_cleaning',   # Glass Cleaner
    'SKU_CLNS005': 'laundry',            # Detergent
    'SKU_CLNS006': 'laundry',            # Washing Soap
    'SKU_CLNS007': 'dish_cleaning',      # Scrub Pad
    'SKU_CLNS008': 'floor_cleaning',     # Mop Set
    'SKU_CLNS009': 'general',            # Garbage Bags
    'SKU_CLNS010': 'general',            # Air Freshener
}

# Product type relationships (more granular than categories)
PRODUCT_TYPE_RELATIONSHIPS = {
    # Cookware needs dish cleaning products
    'cookware': [
        ('dish_cleaning', 0.8),      # ✅ Dishwashing liquid, scrub pads
        ('surface_cleaning', 0.3),   # Maybe glass cleaner for pots
    ],
    
    # Utensils need dish cleaning
    'utensils': [
        ('dish_cleaning', 0.7),
    ],
    
    # Storage containers need cleaning
    'storage': [
        ('dish_cleaning', 0.5),
        ('general', 0.3),            # Garbage bags for food storage
    ],
    
    # Appliances need surface cleaning
    'appliance': [
        ('surface_cleaning', 0.6),
        ('general', 0.4),
    ],
}


def get_product_type(sku: str) -> str:
    """Get the product type for a SKU."""
    return PRODUCT_TYPES.get(sku, 'unknown')


def get_logical_connection_weight(sku1: str, sku2: str) -> float:
    """
    Get the logical connection weight between two specific products.
    
    Returns 0 if no logical connection exists.
    """
    type1 = get_product_type(sku1)
    type2 = get_product_type(sku2)
    
    # Check if type1 has relationship to type2
    if type1 in PRODUCT_TYPE_RELATIONSHIPS:
        for related_type, weight in PRODUCT_TYPE_RELATIONSHIPS[type1]:
            if related_type == type2:
                return weight
    
    # Check reverse (type2 to type1)
    if type2 in PRODUCT_TYPE_RELATIONSHIPS:
        for related_type, weight in PRODUCT_TYPE_RELATIONSHIPS[type2]:
            if related_type == type1:
                return weight
    
    return 0.0
