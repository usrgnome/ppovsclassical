env_config = {
    'max_days': 30,               # Simulate a month of operations
    'initial_inventory': 10,      # Starts with 10 units of milk
    'expiration_days': 2,         # Milk expires quickly (e.g. 2 days)
    'restock_cost': 5,            # Cost to restock one unit (e.g. local dairy logistics)
    'base_demand': 12,            # Reasonable daily demand for a small shop
    'price_range': (1, 10),        # Milk is sold between $1 and $4 per unit
    'max_restock': 10,            # Max order per day (depends on shelf space & supplier)
    'inventory_capacity': 20,     # Small fridge capacity
    'price_levels': 6,            # More granular pricing between 1 and 4
    'restock_levels': 6           # Allows finer restock control (e.g., order 0, 2, 4...10)
}