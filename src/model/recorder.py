from collections import defaultdict
import numpy as np

class EpisodeRecorder:
    def __init__(self):
        self.daily = []
    def log_day(self, env):
        self.daily.append(dict(
            day=env.day,
            price=env.price,
            opening_inventory=env.last_opening_inventory,
            restocked=env.last_restocked,
            expired=env.last_expired,
            visitors=env.last_foot_traffic,
            demand=env.last_sales + env.last_unmet_demand,
            sales=env.last_sales,
            unmet_demand=env.last_unmet_demand,
            revenue=env.last_revenue,
            holding_cost=env.last_holding_cost,
            restock_cost=env.last_restock_cost,
            day_profit=env.last_profit,
            running_profit=env.profit
        ))

def run_policy(env, policy_fn, seed, scenario="Base", policy_name="PPO"):
    state, _ = env.reset(seed=seed)
    rec = EpisodeRecorder()
    while True:
        price, restock = policy_fn(env)  # your policy returns (price, restock)
        _, _, done, truncated, _ = env._step(price, restock)
        rec.log_day(env)
        if done or truncated:
            break
    # aggregate
    d = rec.daily
    arr = lambda k: np.array([r[k] for r in d])
    episode = dict(
        seed=seed, policy=policy_name, scenario=scenario,
        max_days=env.config["max_days"],
        episode_return=float(arr("day_profit").sum()),
        avg_daily_profit=float(arr("day_profit").mean()),
        std_daily_profit=float(arr("day_profit").std(ddof=1)),
        total_revenue=float(arr("revenue").sum()),
        total_restock_cost=float(arr("restock_cost").sum()),
        total_holding_cost=float(arr("holding_cost").sum()),
        stockout_days=int((arr("unmet_demand")>0).sum()),
        total_unmet_demand=int(arr("unmet_demand").sum()),
        pct_unmet_demand=float(arr("unmet_demand").sum()/np.maximum(arr("demand").sum(),1)),
        total_expired=int(arr("expired").sum()),
        price_volatility=float(arr("price").std(ddof=1)),
        restock_frequency=float((arr("restocked")>0).mean()),
        mean_restock_qty=float(arr("restocked")[arr("restocked")>0].mean() if (arr("restocked")>0).any() else 0.0),
        mean_foot_traffic=float(arr("visitors").mean()),
        mean_inventory=float(arr("opening_inventory").mean()),
        max_inventory_utilization=float(arr("opening_inventory").max()/env.config["inventory_capacity"])
    )
    return episode, rec.daily
