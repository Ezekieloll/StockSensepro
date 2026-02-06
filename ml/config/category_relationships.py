"""
Category Relationship Configuration
Defines logical relationships between product categories for demand propagation.

Relationship Types:
- COMPLEMENT: Products used together (e.g., Bread + Butter)
- SUBSTITUTE: Alternative products (e.g., Rice vs Pasta)
- MEAL_COMPONENT: Products used in same meal preparation
- PRICE_SENSITIVE: If category A price ↑, demand for category B ↑
"""

from typing import Dict, List, Set

# Category code to full name mapping
CATEGORY_NAMES = {
    'GROC': 'Grocery (Staples & Grains)',
    'FRPR': 'Fresh Produce (Fruits & Vegetables)',
    'BEVG': 'Beverages',
    'BKDY': 'Bakery & Dairy',
    'FRZN': 'Frozen Foods',
    'SNCK': 'Snacks',
    'MEAT': 'Meat & Seafood',
    'PRSN': 'Personal Care',
    'BABC': 'Baby Care',
    'CLOT': 'Clothing',
    'FTRW': 'Footwear',
    'JWCH': 'Jewelry & Watches',
    'BAGL': 'Bags & Luggage',
    'ELEC': 'Electronics',
    'STOF': 'Stationery & Office',
    'FURH': 'Furniture',
    'BEDM': 'Bedding & Mattress',
    'CLNS': 'Cleaning Supplies',
    'KICH': 'Kitchen Appliances',
    'PETC': 'Pet Care',
    'SPRT': 'Sports Equipment',
    'TOYG': 'Toys & Games',
    'AUTO': 'Automotive',
    'BOOK': 'Books & Media'
}


# =============================================================================
# FOOD & BEVERAGE RELATIONSHIPS
# =============================================================================

# Products that are commonly consumed together (COMPLEMENT relationships)
FOOD_COMPLEMENTS = {
    'GROC': ['BKDY', 'FRPR', 'MEAT', 'BEVG'],  # Rice/Pasta with vegetables, dairy, meat
    'BKDY': ['GROC', 'BEVG', 'SNCK'],          # Bread with butter/milk, beverages
    'FRPR': ['GROC', 'BKDY', 'MEAT'],          # Vegetables with grains, dairy, meat
    'MEAT': ['GROC', 'FRPR', 'FRZN'],          # Meat with rice/pasta, vegetables, frozen sides
    'BEVG': ['GROC', 'BKDY', 'SNCK', 'FRZN'],  # Drinks with food items
    'SNCK': ['BEVG', 'BKDY'],                   # Snacks with drinks, dairy
    'FRZN': ['GROC', 'BEVG', 'MEAT'],          # Frozen foods with staples, drinks
}

# Products that are substitutes (if price of A ↑, demand of B ↑)
FOOD_SUBSTITUTES = {
    'GROC': ['GROC', 'BKDY', 'FRZN'],  # Rice ↔ Pasta ↔ Frozen meals
    'BKDY': ['GROC', 'BKDY'],           # Bread types are substitutes
    'FRPR': ['FRPR', 'FRZN'],           # Fresh vegetables ↔ Frozen vegetables
    'MEAT': ['MEAT', 'FRZN'],           # Fresh meat ↔ Frozen meat
    'BEVG': ['BEVG'],                   # Different beverages substitute each other
    'SNCK': ['SNCK'],                   # Different snacks substitute each other
}


# =============================================================================
# HOUSEHOLD & PERSONAL CARE RELATIONSHIPS
# =============================================================================

HOUSEHOLD_COMPLEMENTS = {
    'PRSN': ['CLNS', 'BABC'],           # Personal care with cleaning, baby items
    'CLNS': ['PRSN', 'KICH'],           # Cleaning with personal care, kitchen
    'KICH': ['GROC', 'CLNS'],           # Kitchen appliances with food, cleaning
    'BABC': ['PRSN', 'CLNS', 'BKDY'],   # Baby care with personal items, cleaning, food
}


# =============================================================================
# CLOTHING & ACCESSORIES RELATIONSHIPS
# =============================================================================

FASHION_COMPLEMENTS = {
    'CLOT': ['FTRW', 'BAGL', 'JWCH'],   # Clothing with shoes, bags, jewelry
    'FTRW': ['CLOT', 'BAGL'],           # Footwear with clothing, bags
    'BAGL': ['CLOT', 'FTRW'],           # Bags with clothing, shoes
    'JWCH': ['CLOT'],                    # Jewelry with clothing
}


# =============================================================================
# HOME & LIFESTYLE RELATIONSHIPS
# =============================================================================

HOME_COMPLEMENTS = {
    'FURH': ['BEDM', 'CLNS', 'KICH'],   # Furniture with bedding, cleaning, kitchen
    'BEDM': ['FURH', 'CLNS'],           # Bedding with furniture, cleaning
}


# =============================================================================
# TECHNOLOGY & ENTERTAINMENT RELATIONSHIPS
# =============================================================================

TECH_COMPLEMENTS = {
    'ELEC': ['STOF', 'BAGL'],           # Electronics with accessories, bags
    'BOOK': ['STOF'],                    # Books with stationery
    'TOYG': ['BOOK', 'BABC'],           # Toys with children's books, baby items
}


# =============================================================================
# SPECIALIZED CATEGORIES
# =============================================================================

SPECIALIZED_COMPLEMENTS = {
    'SPRT': ['CLOT', 'FTRW', 'BEVG'],   # Sports with athletic wear, shoes, energy drinks
    'PETC': [],                          # Pet care is independent
    'AUTO': [],                          # Automotive is independent
}


# =============================================================================
# COMBINED RELATIONSHIP MAPPING
# =============================================================================

def get_category_relationships() -> Dict[str, Dict[str, List[str]]]:
    """
    Get comprehensive category relationships.
    
    Returns:
        Dict with structure:
        {
            'GROC': {
                'complements': ['BKDY', 'FRPR', ...],
                'substitutes': ['GROC', 'BKDY', ...],
                'strength': float  # propagation strength multiplier
            }
        }
    """
    
    relationships = {}
    
    # Combine all complement relationships
    all_complements = {
        **FOOD_COMPLEMENTS,
        **HOUSEHOLD_COMPLEMENTS,
        **FASHION_COMPLEMENTS,
        **HOME_COMPLEMENTS,
        **TECH_COMPLEMENTS,
        **SPECIALIZED_COMPLEMENTS
    }
    
    # Build relationship graph
    for category in CATEGORY_NAMES.keys():
        relationships[category] = {
            'complements': all_complements.get(category, []),
            'substitutes': FOOD_SUBSTITUTES.get(category, [category]),  # At least same category
            'propagation_strength': get_propagation_strength(category)
        }
    
    return relationships


def get_propagation_strength(category: str) -> float:
    """
    Get how strongly a demand change in this category should propagate.
    
    Food categories have stronger propagation (people buy meals together).
    Non-food categories have weaker propagation (more independent purchases).
    """
    
    # High propagation: Food & beverages (bought together in meal planning)
    high_propagation = {'GROC', 'BKDY', 'FRPR', 'MEAT', 'BEVG', 'FRZN', 'SNCK'}
    
    # Medium propagation: Household essentials (bought during same shopping trip)
    medium_propagation = {'PRSN', 'CLNS', 'BABC', 'KICH'}
    
    # Low propagation: Fashion & lifestyle (independent decisions)
    low_propagation = {'CLOT', 'FTRW', 'BAGL', 'JWCH', 'FURH', 'BEDM', 'ELEC', 'BOOK', 'TOYG', 'SPRT', 'STOF'}
    
    # Very low propagation: Specialized categories
    very_low_propagation = {'PETC', 'AUTO'}
    
    if category in high_propagation:
        return 0.8
    elif category in medium_propagation:
        return 0.5
    elif category in low_propagation:
        return 0.3
    else:
        return 0.1


def are_categories_related(cat1: str, cat2: str, relationship_type: str = 'any') -> bool:
    """
    Check if two categories have a meaningful relationship.
    
    Args:
        cat1: First category code
        cat2: Second category code
        relationship_type: 'complement', 'substitute', or 'any'
    
    Returns:
        True if categories are related
    """
    
    if cat1 == cat2:
        return True  # Same category always related
    
    relationships = get_category_relationships()
    
    if cat1 not in relationships or cat2 not in relationships:
        return False
    
    if relationship_type == 'complement' or relationship_type == 'any':
        if cat2 in relationships[cat1]['complements']:
            return True
    
    if relationship_type == 'substitute' or relationship_type == 'any':
        if cat2 in relationships[cat1]['substitutes']:
            return True
    
    return False


def get_propagation_multiplier(source_cat: str, target_cat: str, base_multiplier: float) -> float:
    """
    Calculate how much demand impact should propagate from source to target category.
    
    Args:
        source_cat: Category where demand change originated
        target_cat: Category receiving propagated impact
        base_multiplier: Original demand multiplier (e.g., 1.5 for +50% increase)
    
    Returns:
        Adjusted multiplier for target category
    """
    
    if source_cat == target_cat:
        return base_multiplier  # Full impact on same category
    
    if not are_categories_related(source_cat, target_cat):
        return 1.0  # No impact on unrelated categories
    
    relationships = get_category_relationships()
    
    # Get propagation strength
    strength = relationships[source_cat]['propagation_strength']
    
    # Check relationship type
    is_complement = target_cat in relationships[source_cat].get('complements', [])
    is_substitute = target_cat in relationships[source_cat].get('substitutes', [])
    
    if is_complement:
        # Complements: If source demand ↑, target demand ↑ (proportional)
        # Example: Rice price ↑ → Rice demand ↓ → Vegetables demand stays same or ↓
        # But if event causes Rice demand ↑ → Vegetables demand ↑ (cooking more meals)
        impact = (base_multiplier - 1.0) * strength * 0.7  # 70% of impact
        return 1.0 + impact
    
    elif is_substitute:
        # Substitutes: Complex relationship
        # If price increase causes demand decrease → substitutes see demand increase
        # If event causes demand increase → substitutes see demand stay same or decrease
        # For now, use moderate positive correlation
        impact = (base_multiplier - 1.0) * strength * 0.4  # 40% of impact
        return 1.0 + impact
    
    else:
        return 1.0  # No propagation


# =============================================================================
# TESTING & VALIDATION
# =============================================================================

def print_relationship_summary():
    """Print human-readable summary of all category relationships."""
    
    relationships = get_category_relationships()
    
    print("=" * 80)
    print("CATEGORY RELATIONSHIP SUMMARY")
    print("=" * 80)
    
    for cat_code in sorted(CATEGORY_NAMES.keys()):
        cat_name = CATEGORY_NAMES[cat_code]
        rel = relationships[cat_code]
        
        print(f"\n{cat_code} - {cat_name}")
        print(f"  Propagation Strength: {rel['propagation_strength']:.1f}")
        
        if rel['complements']:
            comp_names = [f"{c} ({CATEGORY_NAMES[c][:20]})" for c in rel['complements']]
            print(f"  Complements: {', '.join(comp_names)}")
        
        if len(rel['substitutes']) > 1 or (len(rel['substitutes']) == 1 and rel['substitutes'][0] != cat_code):
            sub_names = [f"{s}" for s in rel['substitutes'] if s != cat_code]
            if sub_names:
                print(f"  Substitutes: {', '.join(sub_names)}")


if __name__ == "__main__":
    print_relationship_summary()
    
    # Test examples
    print("\n" + "=" * 80)
    print("EXAMPLE PROPAGATION TESTS")
    print("=" * 80)
    
    test_cases = [
        ("GROC", "BKDY", 1.5),  # Rice demand +50%
        ("GROC", "FRPR", 1.5),  # Rice demand +50%
        ("GROC", "CLOT", 1.5),  # Rice demand +50%
        ("BKDY", "BEVG", 1.3),  # Dairy demand +30%
        ("MEAT", "GROC", 1.4),  # Meat demand +40%
    ]
    
    for source, target, multiplier in test_cases:
        result = get_propagation_multiplier(source, target, multiplier)
        impact_pct = (result - 1.0) * 100
        print(f"\n{source} → {target}: {multiplier:.2f}x ({(multiplier-1)*100:+.0f}%)")
        print(f"  → Propagated: {result:.2f}x ({impact_pct:+.1f}%)")
        print(f"  → Related: {are_categories_related(source, target)}")
