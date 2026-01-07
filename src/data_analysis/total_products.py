"""
Counts and prints the number of distinct products, categories, and brands.
- Uses movimentos_saida_mercadoria (PRODUCT, PRODUCTHIERARCHY1/2/3, BRAND)
- Prints counts for each field
"""

from utils.data_split import movimentos_saida_mercadoria

# Count and print the number of unique products
products = movimentos_saida_mercadoria['PRODUCT'].nunique()
print(f"Number of distinct products: {products}")

# Count and print the number of unique categories in hierarchy 1
product_categories_h1 = movimentos_saida_mercadoria['PRODUCTHIERARCHY1'].nunique()
print(f"Number of distinct categories (Hierarchy 1): {product_categories_h1}")

# Count and print the number of unique categories in hierarchy 2
product_categories_h2 = movimentos_saida_mercadoria['PRODUCTHIERARCHY2'].nunique()
print(f"Number of distinct categories (Hierarchy 2): {product_categories_h2}")

# Count and print the number of unique categories in hierarchy 3
product_categories_h3 = movimentos_saida_mercadoria['PRODUCTHIERARCHY3'].nunique()
print(f"Number of distinct categories (Hierarchy 3): {product_categories_h3}")

# Count and print the number of unique brands
brands = movimentos_saida_mercadoria['BRAND'].nunique()
print(f"Number of distinct brands: {brands}")