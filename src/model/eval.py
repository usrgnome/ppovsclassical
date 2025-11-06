# evaluator.py
import argparse
import json
import math
from statistics import NormalDist
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from ql_wrapper import InventoryPPOEnv

# 3) Pretty boxed console table (no deps)
import sys

def _fmt_num(v):
    if isinstance(v, (int, float, np.floating)) and pd.notna(v):
        return f"{v:,.2f}"
    return "" if (isinstance(v, float) and pd.isna(v)) else str(v)

def print_boxed_table(df: pd.DataFrame, left_align=("policy",), ascii_only: bool | None = None):
    cols = list(df.columns)
    rows = []
    for _, r in df.iterrows():
        rows.append([_fmt_num(r[c]) if c != "policy" else str(r[c]) for c in cols])

    # Decide whether to use Unicode or ASCII
    if ascii_only is None:
        try:
            "┌┬┐└┴┘├┼┤│─".encode(sys.stdout.encoding or "utf-8")
            use_unicode = True
        except Exception:
            use_unicode = False
    else:
        use_unicode = not ascii_only

    if use_unicode:
        H, V = "─", "│"
        TL, TJ, TR = "┌", "┬", "┐"
        ML, MJ, MR = "├", "┼", "┤"
        BL, BJ, BR = "└", "┴", "┘"
    else:
        H, V = "-", "|"
        TL, TJ, TR = "+", "+", "+"
        ML, MJ, MR = "+", "+", "+"
        BL, BJ, BR = "+", "+", "+"

    # Compute column widths
    widths = []
    for i, c in enumerate(cols):
        col_vals = [str(x) for x in [c] + [row[i] for row in rows]]
        widths.append(max(len(x) for x in col_vals))

    def horiz(left, join, right):
        return left + join.join(H * (w + 2) for w in widths) + right

    def make_row(values):
        cells = []
        for i, s in enumerate(values):
            w = widths[i]
            if cols[i] in left_align:
                cells.append(" " + str(s).ljust(w) + " ")
            else:
                cells.append(" " + str(s).rjust(w) + " ")
        return V + V.join(cells) + V

    # Print table
    print(horiz(TL, TJ, TR))
    print(make_row(cols))
    print(horiz(ML, MJ, MR))
    for r in rows:
        print(make_row(r))
    print(horiz(BL, BJ, BR))

# ===== Default config (mirrors your training BASE_CONFIG) =====
DEFAULT_BASE_CONFIG2 = dict(
    max_days=30,
    initial_inventory=30,
    expiration_days=3,
    restock_cost=1.2,
    holding_cost=0.5,
    waste_cost=0.3,
    base_demand=50,
    price_range=(1.0, 6.0),
    max_restock=15,
    inventory_capacity=30,
    initial_price=3.5,
    demand_config=dict(
        traffic=dict(
            base=50,
            weekdayMultipliers=[0.9,0.9,1.0,1.0,1.1,1.3,1.4],
            weekdayStart="Mon",
            visitorNoiseStd=40.0,
            priceTrafficCut=False,
            priceTrafficDelta=15.0,
        ),
        conversion=dict(
            model="wtpLogNormal",
            p1=3.0, c1=0.15,
            p2=5.0, c2=0.08,
            p0=10.0, c0=0.20, elasticity=-0.5,
            linearPriceCoeff=1.0,
            expElasticity=0.05,
        ),
        noise=dict(model="poisson", negbinK=5.0),
    ),
)

DEFAULT_BASE_CONFIG = dict(
    max_days=30,
    initial_inventory=30,
    expiration_days=6,
    restock_cost=2.0,
    holding_cost=0.5,
    waste_cost=0.3,
    base_demand=50,
    price_range=(1.0, 12.0),
    max_restock=15,
    inventory_capacity=30,
    initial_price=3.5,
    demand_config=dict(
        traffic=dict(
            base=50,
            weekdayMultipliers=[0.9,0.9,1.0,1.0,1.1,1.3,1.4],
            weekdayStart="Mon",
            visitorNoiseStd=40.0,
            priceTrafficCut=False,
            priceTrafficDelta=15.0,
        ),
        conversion=dict(
            model="wtpLogNormal",
            p1=6.0, c1=0.15,
            p2=10.0, c2=0.08,
            p0=10.0, c0=0.20, elasticity=-0.5,
            linearPriceCoeff=1.0,
            expElasticity=0.05,
        ),
        noise=dict(model="poisson", negbinK=5.0),
    ),
)

# ===== Helpers: deterministic expectation given today's traffic + conversion model =====
def _traffic_multiplier_from_price(traffic_cfg: Dict, price: float, price_range: Tuple[float, float]) -> float:
    # mirrors InventoryPricingEnv._price_to_visitors_multiplier
    if not traffic_cfg.get("priceTrafficCut", False):
        return 1.0
    pivot = float(traffic_cfg.get("priceTrafficDelta", price_range[1]))
    if price <= pivot:
        return 1.0
    ratio = price / max(1e-9, pivot)
    mult = 1.0 / (1.0 + (ratio - 1.0))
    return float(max(0.1, min(1.0, mult)))

def _conv_prob(conv_cfg: Dict, price_range: Tuple[float, float], p: float) -> float:
    # mirrors InventoryPricingEnv._price_to_conversion
    p = max(float(p), 1e-9)
    model = str(conv_cfg.get("model", "wtpLogNormal"))

    if model == "wtpLogNormal":
        nd = NormalDist()
        p1 = float(conv_cfg.get("p1", 5.0)); c1 = _clamp01(conv_cfg.get("c1", 0.40))
        p2 = float(conv_cfg.get("p2", 15.0)); c2 = _clamp01(conv_cfg.get("c2", 0.10))
        z1 = nd.inv_cdf(1.0 - c1); z2 = nd.inv_cdf(1.0 - c2)
        denom = (z2 - z1) or 1e-9
        sigma = (math.log(p2) - math.log(p1)) / denom
        mu = math.log(p1) - sigma * z1
        x = (math.log(p) - mu) / sigma
        conv = 1.0 - nd.cdf(x)

    elif model == "logitLogPrice":
        def logit(x: float) -> float:
            x = _clamp01(x); return math.log(x / (1.0 - x + 1e-12))
        p1 = float(conv_cfg.get("p1", 5.0)); c1 = _clamp01(conv_cfg.get("c1", 0.40))
        p2 = float(conv_cfg.get("p2", 15.0)); c2 = _clamp01(conv_cfg.get("c2", 0.10))
        x1, x2 = math.log(p1), math.log(p2)
        L1, L2 = logit(c1), logit(c2)
        beta = -(L2 - L1) / (x2 - x1 + 1e-12)
        alpha = -L1 - beta * x1
        x = math.log(p)
        conv = 1.0 / (1.0 + math.exp(alpha + beta * x))

    elif model == "elasticity":
        p0 = float(conv_cfg.get("p0", 10.0))
        c0 = _clamp01(conv_cfg.get("c0", 0.20))
        eps = float(conv_cfg.get("elasticity", -1.5))
        conv = c0 * (p / max(1e-9, p0)) ** eps

    elif model == "linear":
        lo, hi = price_range
        k = float(conv_cfg.get("linearPriceCoeff", 1.0))
        conv = 1.0 - k * ((p - lo) / max(1e-9, (hi - lo)))

    elif model == "exponential":
        alpha = float(conv_cfg.get("expElasticity", 0.05))
        conv = math.exp(-alpha * p)

    else:
        conv = 0.0

    return float(_clamp(conv, 0.0, 1.0))

def expected_visitors_today(env: InventoryPPOEnv, price: float) -> int:
    g = env.env
    base = int(g._visitors_for_day(g.day))  # deterministic precomputed visitors
    mult = _traffic_multiplier_from_price(g.traffic_cfg, price, g.config["price_range"])
    return max(0, int(round(base * mult)))

def expected_demand_today(env: InventoryPPOEnv, price: float) -> float:
    g = env.env
    conv = _conv_prob(g.conv_cfg, g.config["price_range"], price)
    return float(expected_visitors_today(env, price) * conv)

# ===== Policy wrappers =====
class Policy:
    name: str
    def act(self, obs: np.ndarray, env: InventoryPPOEnv) -> np.ndarray:  # returns [price_abs, restock_frac]
        raise NotImplementedError

class PPOPolicy(Policy):
    def __init__(self, model_path: str,
                 seed: int = 0):
        vecnorm_path = "vecnorm.pkl"
        self.name = "ppo"
        self.model = PPO.load(model_path)

        # Optional VecNormalize support
        self.vecnorm = None
        if vecnorm_path:
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from stable_baselines3.common.monitor import Monitor

            # make a 1-env dummy only to load stats (we won't step it)
            def make_env():
                return Monitor(InventoryPPOEnv(DEFAULT_BASE_CONFIG,
                                               seed=seed, render_mode="bot"))
            dummy = DummyVecEnv([make_env])
            self.vecnorm = VecNormalize.load(vecnorm_path, dummy)
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False

    def act(self, obs: np.ndarray, env: InventoryPPOEnv) -> np.ndarray:
        # normalize obs if vecnorm stats available
        if self.vecnorm is not None:
            o = self.vecnorm.normalize_obs(obs[None, :])
            o = o[0]
        else:
            o = obs

        action, _ = self.model.predict(o, deterministic=True)
        return np.asarray(action, dtype=np.float32)

class FixedPricePolicy(Policy):
    """
    Keeps price fixed; restocks aggressively (fill-to-cap) each day.
    """
    def __init__(self, price: float):
        self.price = float(price)
        self.name = f"fixed_price_{self.price:g}"

    def act(self, obs: np.ndarray, env: InventoryPPOEnv) -> np.ndarray:
        lo, hi = env._price_lo, env._price_hi
        price = float(np.clip(self.price, lo, hi))
        restock_frac = 1.0
        return np.array([price, restock_frac], dtype=np.float32)

class MyopicGreedyPolicy(Policy):
    """
    Each day, choose price (grid) that maximizes expected same-day profit:
      E[profit] = price * E[sales] - restock_cost * restock_units
    where E[sales] = min(E[demand(price)], opening_inventory + restock_units).
    Restock just enough to (approximately) cover E[demand], bounded by max_restock and capacity.
    """
    def __init__(self, step: float = 0.1):
        self.name = f"myopic_greedy_step{step:g}"
        self.step = max(1e-3, float(step))

    def act(self, obs: np.ndarray, env: InventoryPPOEnv) -> np.ndarray:
        g = env.env
        lo, hi = env._price_lo, env._price_hi
        cap = env._cap
        max_restock = env._max_restock
        restock_cost = float(g.config["restock_cost"])

        opening_inv = int(g.inventory)
        cap_left = max(0, cap - opening_inv)

        best_price = lo
        best_profit = -1e18
        best_restock_units = 0

        n_steps = max(2, int(round((hi - lo) / self.step)) + 1)
        prices = np.linspace(lo, hi, n_steps)

        for p in prices:
            dem = expected_demand_today(env, float(p))
            need = max(0.0, dem - opening_inv)
            restock_units = int(min(max_restock, cap_left, math.ceil(need)))
            exp_sales = min(opening_inv + restock_units, dem)
            exp_profit = float(p) * exp_sales - restock_cost * restock_units
            if exp_profit > best_profit:
                best_profit = exp_profit
                best_price = float(p)
                best_restock_units = restock_units

        restock_frac = float(best_restock_units) / float(max_restock) if max_restock > 0 else 0.0
        restock_frac = float(np.clip(restock_frac, 0.0, 1.0))
        return np.array([best_price, restock_frac], dtype=np.float32)

class ThresholdRestockPolicy(Policy):
    """
    Classic (s, S) policy with fixed price:
      - If inventory < s, restock up to S (capped by capacity and max_restock).
      - Else, no restock.
    """
    def __init__(self, price: float, s_level: int, S_level: int):
        self.price = float(price)
        self.s = int(s_level)
        self.S = int(S_level)
        self.name = f"sS_price{self.price:g}_s{self.s}_S{self.S}"

    def act(self, obs: np.ndarray, env: InventoryPPOEnv) -> np.ndarray:
        g = env.env
        lo, hi = env._price_lo, env._price_hi
        price = float(np.clip(self.price, lo, hi))

        inv = int(g.inventory)
        cap_left = max(0, env._cap - inv)
        max_restock = env._max_restock

        restock_units = 0
        if inv < self.s:
            target = max(0, self.S - inv)
            restock_units = min(target, cap_left, max_restock)

        restock_frac = float(restock_units) / float(max_restock) if max_restock > 0 else 0.0
        restock_frac = float(np.clip(restock_frac, 0.0, 1.0))
        return np.array([price, restock_frac], dtype=np.float32)

class MarkdownExpiryPolicy(Policy):
    """
    Heuristic dynamic pricing sensitive to inventory/expiry:
      price_t = clip(
          p0 + k_inv*(inv_frac_target - inv_frac) - k_exp*expiry_pressure + k_dow*sin_dow,
          [lo, hi]
      )
    Restock to keep inventory near 'target_inv' each day (bounded).
    """
    def __init__(self, base_price: float, target_inv_frac: float = 0.5,
                 k_inv: float = 1.0, k_exp: float = 1.0, k_dow: float = 0.0,
                 target_inv_units: Optional[int] = None):
        self.p0 = float(base_price)
        self.tgt_frac = float(np.clip(target_inv_frac, 0.0, 1.0))
        self.k_inv = float(k_inv)
        self.k_exp = float(k_exp)
        self.k_dow = float(k_dow)
        self.target_inv_units = target_inv_units  # if provided, overrides frac
        self.name = f"markdown_p0{self.p0:g}_t{self.tgt_frac:g}"

    def act(self, obs: np.ndarray, env: InventoryPPOEnv) -> np.ndarray:
        inv_frac       = float(obs[0])
        dow_sin        = float(obs[2])
        expiry_press   = float(obs[8])

        lo, hi = env._price_lo, env._price_hi
        cap = env._cap
        max_restock = env._max_restock
        g = env.env

        delta = self.k_inv * (self.tgt_frac - inv_frac) - self.k_exp * expiry_press + self.k_dow * dow_sin
        price = float(np.clip(self.p0 + delta, lo, hi))

        target_units = self.target_inv_units if self.target_inv_units is not None else int(round(self.tgt_frac * cap))
        inv = int(g.inventory)
        need = max(0, target_units - inv)
        cap_left = max(0, cap - inv)
        restock_units = min(need, cap_left, max_restock)

        restock_frac = float(restock_units) / float(max_restock) if max_restock > 0 else 0.0
        restock_frac = float(np.clip(restock_frac, 0.0, 1.0))
        return np.array([price, restock_frac], dtype=np.float32)

class SellThroughRHPolicy(Policy):
    """
    Rolling-horizon sell-through (MPC-lite).
    """
    def __init__(
        self,
        horizon_days: int = 3,
        target_doc_days: float = 2.0,
        safety_frac: float = 0.10,
        price_step: float = 0.05,
        min_price_move: float = 0.0,
    ):
        self.name = f"sellthrough_rh_H{horizon_days}_doc{target_doc_days:g}"
        self.H = max(1, int(horizon_days))
        self.doc = max(0.1, float(target_doc_days))
        self.safety = max(0.0, float(safety_frac))
        self.dp = max(1e-3, float(price_step))
        self.min_move = max(0.0, float(min_price_move))
        self._last_price = None

    def _inverse_demand_today(self, env: InventoryPPOEnv, target_sales: float) -> float:
        lo, hi = env._price_lo, env._price_hi
        steps = max(2, int(round((hi - lo) / self.dp)) + 1)
        grid = np.linspace(lo, hi, steps)
        best_p = lo
        best_err = float("inf")
        for p in grid:
            dem = expected_demand_today(env, float(p))
            err = abs(dem - target_sales)
            if err < best_err:
                best_err = err
                best_p = float(p)
        return best_p

    def _expected_demand_next_days(self, env: InventoryPPOEnv, p: float, days: int) -> float:
        g = env.env
        total = 0.0
        for h in range(days):
            base = int(g._visitors_for_day(g.day + h))
            mult = _traffic_multiplier_from_price(g.traffic_cfg, p, g.config["price_range"])
            conv = _conv_prob(g.conv_cfg, g.config["price_range"], p)
            total += float(int(round(base * mult))) * float(conv)
        return total

    def act(self, obs: np.ndarray, env: InventoryPPOEnv) -> np.ndarray:
        g = env.env
        lo, hi = env._price_lo, env._price_hi
        cap, max_restock = env._cap, env._max_restock

        inv = int(g.inventory)
        cap_left = max(0, cap - inv)
        E = int(g.config.get("expiration_days", 3))
        days_left = max(1, min(E, int(g.config["max_days"] - g.day)))

        H = min(self.H, days_left)
        mid_price = 0.5 * (lo + hi)
        mu_today_mid = expected_demand_today(env, mid_price)
        target_eod_inv_today = max(0.0, inv - (inv / max(1, days_left)))
        doc_target = max(0.0, self.doc * mu_today_mid)
        target_eod_inv_today = max(target_eod_inv_today, doc_target * 0.5)

        target_sales_no_r = max(0.0, inv - target_eod_inv_today)
        target_sales_no_r *= (1.0 + self.safety)

        p_raw = self._inverse_demand_today(env, target_sales_no_r)
        if self._last_price is not None and self.min_move > 0.0:
            delta = np.clip(p_raw - self._last_price, -self.min_move, self.min_move)
            p = float(np.clip(self._last_price + delta, lo, hi))
        else:
            p = float(np.clip(p_raw, lo, hi))

        mu_today = expected_demand_today(env, p)
        mu_doc = self._expected_demand_next_days(env, p, int(math.ceil(self.doc)))
        desired_post_restock = mu_doc
        need = max(0.0, desired_post_restock - inv)
        restock_units = int(min(max_restock, cap_left, math.ceil(need)))
        restock_frac = float(restock_units) / float(max_restock) if max_restock > 0 else 0.0
        restock_frac = float(np.clip(restock_frac, 0.0, 1.0))

        self._last_price = p
        return np.array([p, restock_frac], dtype=np.float32)

class NewsvendorPolicy(Policy):
    """
    Single-period newsvendor with price selection.
    """
    def __init__(
        self,
        price_step: float = 0.1,
        cover_days: float = 1.0,
        overage_cost_extra_waste: float = 0.0,
        min_service_level: float = 0.5,
        max_service_level: float = 0.99,
    ):
        self.name = f"newsvendor_step{price_step:g}"
        self.price_step = max(1e-3, float(price_step))
        self.cover_days = max(1e-3, float(cover_days))
        self.overage_cost_extra_waste = max(0.0, float(overage_cost_extra_waste))
        self.min_SL = float(min_service_level)
        self.max_SL = float(max_service_level)

    def _demand_stats(self, env: InventoryPPOEnv, p: float) -> tuple[float, float]:
        g = env.env
        mu1 = float(expected_demand_today(env, p))
        mu = mu1 * self.cover_days

        noise_cfg = (g.config.get("demand_config") or {}).get("noise", {})
        model = str(noise_cfg.get("model", "poisson")).lower()

        if model == "poisson":
            var = max(1.0, mu)
        elif model in ("negbin", "negativebinomial", "neg_bin"):
            k = float(noise_cfg.get("negbinK", 5.0))
            var = max(1.0, mu + (mu * mu) / max(1e-9, k))
        else:
            var = max(1.0, mu * 1.2)
        return mu, var

    def _q_from_service_level(self, mu: float, var: float, SL: float) -> float:
        SL = float(min(self.max_SL, max(self.min_SL, SL)))
        z = NormalDist().inv_cdf(SL)
        return mu + z * math.sqrt(max(1e-9, var))

    def act(self, obs: np.ndarray, env: InventoryPPOEnv) -> np.ndarray:
        g = env.env
        lo, hi = env._price_lo, env._price_hi
        cap, max_restock = env._cap, env._max_restock

        opening_inv = int(g.inventory)
        cap_left = max(0, cap - opening_inv)

        restock_cost = float(g.config["restock_cost"])
        holding_cost = float(g.config.get("holding_cost", 0.0))
        stockout_penalty = float(getattr(env, "_stockout_penalty", 0.0))  # FIXED: use wrapper attribute

        n_steps = max(2, int(round((hi - lo) / self.price_step)) + 1)
        prices = np.linspace(lo, hi, n_steps)

        best_profit = -1e18
        best_price = lo
        best_restock_units = 0

        for p in prices:
            mu, var = self._demand_stats(env, float(p))

            margin = max(0.0, float(p) - restock_cost)
            Cu = margin + stockout_penalty
            Co = holding_cost + self.overage_cost_extra_waste

            SL = Cu / max(1e-9, (Cu + Co))
            Q_star = self._q_from_service_level(mu, var, SL)

            target = max(0.0, Q_star)
            need = max(0.0, target - opening_inv)
            restock_units = int(min(max_restock, cap_left, math.ceil(need)))

            stock_after = opening_inv + restock_units
            exp_sales = min(stock_after, mu)
            exp_leftover = max(0.0, stock_after - exp_sales)
            exp_stockout = max(0.0, mu - stock_after)

            exp_profit = (
                float(p) * exp_sales
                - restock_cost * restock_units
                - stockout_penalty * exp_stockout
                - holding_cost * exp_leftover
            )

            if exp_profit > best_profit:
                best_profit = exp_profit
                best_price = float(p)
                best_restock_units = restock_units

        restock_frac = float(best_restock_units) / float(max_restock) if max_restock > 0 else 0.0
        return np.array([best_price, float(np.clip(restock_frac, 0.0, 1.0))], dtype=np.float32)

# ===== Evaluation loop =====
def run_episode(env: InventoryPPOEnv, policy: Policy, seed: int, episode_idx: int) -> List[Dict]:
    rows: List[Dict] = []

    obs, _ = env.reset(seed=seed)
    done = False

    # running totals (per episode)
    cum_waste = 0
    cum_missed = 0

    while not done:
        action = policy.act(obs, env)
        obs, reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)

        g = env.env  # underlying game for metrics (InventoryPricingEnv)

        # daily waste & missed sales (units)
        waste_units = int(g.last_expired)
        missed_sales_units = int(g.last_unmet_demand)

        # dollar implications
        restock_cost = float(g.config["restock_cost"])
        price = float(g.price)
        margin = max(0.0, price - restock_cost)
        waste_value_at_cost = float(waste_units) * float(g.config["waste_cost"]) 
        missed_revenue = float(missed_sales_units) * price
        missed_margin = float(missed_sales_units) * margin

        # update running totals
        cum_waste += waste_units
        cum_missed += missed_sales_units

        row = dict(
            policy=policy.name,
            episode=episode_idx,
            day=int(g.day - 1),
            action_price=float(action[0]),
            action_restock_frac=float(action[1]),
            price=price,
            opening_inventory=int(g.last_opening_inventory),
            restocked=int(g.last_restocked),
            expired=waste_units,
            visitors=int(g.last_foot_traffic),
            sales=int(g.last_sales),
            unmet_demand=missed_sales_units,
            closing_inventory=int(g.inventory),
            revenue=float(g.last_revenue),
            restock_cost=float(g.last_restock_cost),
            holding_cost=float(g.last_holding_cost),
            day_profit=float(g.last_profit),
            cum_profit=float(g.profit),
            waste_units=waste_units,
            waste_value_at_cost=waste_value_at_cost,
            missed_sales_units=missed_sales_units,
            missed_revenue=missed_revenue,
            missed_margin=missed_margin,
            cum_waste_units=cum_waste,
            cum_missed_sales_units=cum_missed,
            stockout=int(g.last_unmet_demand > 0),
            capacity=int(env._cap),
            capacity_left_opening=int(max(0, env._cap - g.last_opening_inventory)),
        )
        rows.append(row)

    return rows

def evaluate(
    model_path: str,
    out_csv: str,
    episodes: int,
    base_seed: int,
    fixed_price: float,
    base_config: Dict,
    include_myopic: bool,
    myopic_step: float,
    include_sS: bool,
    s_level: int,
    S_level: int,
    include_markdown: bool,
    markdown_k_inv: float,
    markdown_k_exp: float,
    markdown_k_dow: float,
    markdown_target_frac: float,
    include_newsvendor: bool,
    nv_price_step: float,
    nv_cover_days: float,
    nv_overage_extra: float,
    nv_min_sl: float,
    nv_max_sl: float,
    include_sellthrough: bool,
    st_horizon: float,
    st_doc: float,
    st_safety: float,
    st_price_step: float,
    st_min_move: float,
) -> Tuple[pd.DataFrame, Path]:
    # policies to compare
    bottom, top = DEFAULT_BASE_CONFIG["price_range"]
    middle = (bottom + top) / 2
    
    policies: List[Policy] = [
        #PPOPolicy(model_path),
        FixedPricePolicy(price=fixed_price),  # use config-driven fixed price
        FixedPricePolicy(price=top),
        FixedPricePolicy(price=middle),
        FixedPricePolicy(price=bottom),
    ]

    if include_myopic:
        policies.append(MyopicGreedyPolicy(step=myopic_step))

    if include_sS:
        # also include an sS at the fixed/base price
        policies.append(ThresholdRestockPolicy(price=fixed_price, s_level=s_level, S_level=S_level))
        # and optionally at extremes for comparison
        policies.append(ThresholdRestockPolicy(price=top, s_level=s_level, S_level=S_level))
        policies.append(ThresholdRestockPolicy(price=bottom, s_level=s_level, S_level=S_level))

    if include_markdown:
        policies.append(
            MarkdownExpiryPolicy(
                base_price=fixed_price,
                target_inv_frac=markdown_target_frac,
                k_inv=markdown_k_inv,
                k_exp=markdown_k_exp,
                k_dow=markdown_k_dow,
            )
        )

    if include_newsvendor:
        policies.append(
            NewsvendorPolicy(
                price_step=nv_price_step,
                cover_days=nv_cover_days,
                overage_cost_extra_waste=nv_overage_extra,
                min_service_level=nv_min_sl,
                max_service_level=nv_max_sl,
            )
        )

    if include_sellthrough:
        policies.append(
            SellThroughRHPolicy(
                horizon_days=st_horizon,
                target_doc_days=st_doc,
                safety_frac=st_safety,
                price_step=st_price_step,
                min_price_move=st_min_move,
            )
        )

    # reuse the same seeds for each policy so comparisons are apples-to-apples
    episode_seeds = [base_seed + i for i in range(episodes)]

    all_rows: List[Dict] = []

    for policy in policies:
        for ep_idx, seed in enumerate(episode_seeds, start=1):
            env = InventoryPPOEnv(
                base_config=base_config,
                seed=seed,
                price_jump_penalty=0.0,
                stockout_penalty=0.1,  # same as your train script
                render_mode="bot",      # quiet
            )
            rows = run_episode(env, policy, seed=seed, episode_idx=ep_idx)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.sort_values(by=["policy", "episode", "day"], inplace=True)

    out_path = Path(out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df, out_path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate PPO vs baseline policies and save per-day results to CSV."
    )
    # core
    p.add_argument("--model-path", type=str, default="./ppo_inventory_policy",
                   help="Path to PPO .zip (or base name) saved by SB3.")
    p.add_argument("--out-csv", type=str, default="./eval_daily_results.csv",
                   help="Destination CSV for per-day results.")
    p.add_argument("--episodes", type=int, default=5,
                   help="Episodes per policy (uses same seeds for fairness).")
    p.add_argument("--seed", type=int, default=12345,
                   help="Base seed; each episode uses seed+i.")
    # Make fixed-price default derived from config if omitted
    p.add_argument("--fixed-price", type=float, default=None,
                   help="If omitted, uses config['initial_price'].")

    # config override
    p.add_argument("--config-json", type=str, default="",
                   help="Optional path to a JSON file overriding the base_config used in env.")

    # Optional baselines toggles & params
    p.add_argument("--include-myopic", action="store_true",
                   help="Include MyopicGreedyPolicy baseline.")
    p.add_argument("--myopic-step", type=float, default=0.1,
                   help="Price grid step for myopic greedy (e.g., 0.1).")

    p.add_argument("--include-sS", action="store_true",
                   help="Include ThresholdRestockPolicy (s, S) with fixed price.")
    # Make s/S defaults derived from config if omitted
    p.add_argument("--s-level", type=int, default=None,
                   help="If omitted, derived from config/demand.")
    p.add_argument("--S-level", type=int, default=None,
                   help="If omitted, set to capacity from config.")

    p.add_argument("--include-markdown", action="store_true",
                   help="Include MarkdownExpiryPolicy baseline.")
    p.add_argument("--markdown-target-frac", type=float, default=0.5,
                   help="Target inventory as fraction of capacity (0..1).")
    p.add_argument("--markdown-k-inv", type=float, default=1.0,
                   help="Price sensitivity to (target_inv_frac - inv_frac).")
    p.add_argument("--markdown-k-exp", type=float, default=1.0,
                   help="Price markdown per unit of expiry pressure.")
    p.add_argument("--markdown-k-dow", type=float, default=0.0,
                   help="Optional weekday sinusoidal adjustment strength.")

    # Newsvendor policy toggles
    p.add_argument("--include-newsvendor", action="store_true",
                   help="Include NewsvendorPolicy baseline.")
    p.add_argument("--nv-price-step", type=float, default=0.1,
                   help="Price grid step for newsvendor.")
    p.add_argument("--nv-cover-days", type=float, default=1.0,
                   help="Cover this many days of demand in Q* (hedge).")
    p.add_argument("--nv-overage-extra", type=float, default=0.0,
                   help="Extra overage cost per leftover unit (e.g., perish risk at cost).")
    p.add_argument("--nv-min-sl", type=float, default=0.50,
                   help="Lower clamp for service level.")
    p.add_argument("--nv-max-sl", type=float, default=0.99,
                   help="Upper clamp for service level.")

    p.add_argument("--include-sellthrough", action="store_true",
               help="Include SellThroughRHPolicy (rolling-horizon sell-through) baseline.")
    p.add_argument("--st-horizon", type=int, default=3)
    p.add_argument("--st-doc", type=float, default=2.0)
    p.add_argument("--st-safety", type=float, default=0.10)
    p.add_argument("--st-price-step", type=float, default=0.05)
    p.add_argument("--st-min-move", type=float, default=0.0)
    return p.parse_args()

def load_config(json_path: str | None) -> Dict:
    if json_path:
        cfg_path = Path(json_path)
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(DEFAULT_BASE_CONFIG)
        merged.update(data or {})
        return merged
    return dict(DEFAULT_BASE_CONFIG)

def derive_ss_defaults(base_config: Dict, seed: int) -> tuple[int, int]:
    """
    Compute (s, S) from config:
      - S = capacity
      - s ≈ cover_days * E[demand at initial_price]
      with cover_days = min(2, expiration_days).
    """
    cap = int(base_config["inventory_capacity"])
    max_restock = int(base_config["max_restock"])
    initial_price = float(
        base_config.get("initial_price",
                        0.5 * (base_config["price_range"][0] + base_config["price_range"][1]))
    )

    # small env to estimate demand deterministically via helpers
    tmp_env = InventoryPPOEnv(base_config=base_config, seed=seed, render_mode="bot")
    mu = expected_demand_today(tmp_env, initial_price)

    cover_days = max(1, min(2, int(base_config.get("expiration_days", 3))))
    s = int(np.clip(cover_days * mu, 1, cap))
    S = cap
    # ensure one order can make progress (optional)
    s = min(s, max(0, S - max_restock))
    return s, S

def main():
    args = parse_args()
    base_config = load_config(args.config_json)

    # fixed price default from config if not provided
    fixed_price = args.fixed_price if args.fixed_price is not None else float(base_config["initial_price"])

    # derive (s, S) if not provided
    if args.s_level is None or args.S_level is None:
        s_auto, S_auto = derive_ss_defaults(base_config, seed=args.seed)
        s_level = s_auto if args.s_level is None else int(args.s_level)
        S_level = S_auto if args.S_level is None else int(args.S_level)
    else:
        s_level, S_level = int(args.s_level), int(args.S_level)

    df, out_path = evaluate(
        model_path=args.model_path,
        out_csv=args.out_csv,
        episodes=args.episodes,
        base_seed=args.seed,
        fixed_price=fixed_price,
        base_config=base_config,
        include_myopic=args.include_myopic,
        myopic_step=args.myopic_step,
        include_sS=args.include_sS,
        s_level=s_level,
        S_level=S_level,
        include_markdown=args.include_markdown,
        markdown_k_inv=args.markdown_k_inv,
        markdown_k_exp=args.markdown_k_exp,
        markdown_k_dow=args.markdown_k_dow,
        markdown_target_frac=args.markdown_target_frac,
        include_newsvendor=args.include_newsvendor,
        nv_price_step=args.nv_price_step,
        nv_cover_days=args.nv_cover_days,
        nv_overage_extra=args.nv_overage_extra,
        nv_min_sl=args.nv_min_sl,
        nv_max_sl=args.nv_max_sl,
        include_sellthrough=args.include_sellthrough,
        st_horizon=args.st_horizon,
        st_doc=args.st_doc,
        st_safety=args.st_safety,
        st_price_step=args.st_price_step,
        st_min_move=args.st_min_move,
    )

    # quick console summary (averaged per policy across episodes)
    # 1) roll up per-episode totals
    per_episode = (
        df.groupby(["policy", "episode"])
          .agg(
              final_cum_profit=("cum_profit", "last"),
              total_sales=("sales", "sum"),
              total_unmet=("missed_sales_units", "sum"),
              total_waste=("waste_units", "sum"),
              missed_margin_total=("missed_margin", "sum"),
              waste_cost_total=("waste_value_at_cost", "sum"),
          )
          .reset_index()
    )

    # 2) average across episodes per policy (plus std for reference)
    summary = (
        per_episode.groupby("policy")
                   .agg(
                       final_cum_profit_mean=("final_cum_profit", "mean"),
                       final_cum_profit_std=("final_cum_profit", "std"),
                       total_sales_mean=("total_sales", "mean"),
                       total_unmet_mean=("total_unmet", "mean"),
                       total_waste_mean=("total_waste", "mean"),
                       missed_margin_mean=("missed_margin_total", "mean"),
                       waste_cost_mean=("waste_cost_total", "mean"),
                       episodes=("episode", "nunique"),
                   )
                   .reset_index()
                   .sort_values("final_cum_profit_mean", ascending=False)
    )

    # ---- add simple uncertainty on final_cum_profit_mean ----
    # Standard error and 95% normal-approx CI (episodes is small-ish but OK; swap for t if you prefer)
    summary["profit_se"] = summary["final_cum_profit_std"] / np.sqrt(summary["episodes"])
    z = NormalDist().inv_cdf(1 - 0.05/2)   # 1.96 for 95%
    summary["profit_moe"] = z * summary["profit_se"]
    summary["profit_ci_low"] = summary["final_cum_profit_mean"] - summary["profit_moe"]
    summary["profit_ci_high"] = summary["final_cum_profit_mean"] + summary["profit_moe"]
    summary["final_cum_profit_95ci"] = summary.apply(
        lambda r: f"[{r['profit_ci_low']:.2f}, {r['profit_ci_high']:.2f}]",
        axis=1,
    )

    # Optional: choose compact columns for printing
    display_cols = [
        "policy",
        "final_cum_profit_mean",
        "total_sales_mean",
        "total_unmet_mean",
        "total_waste_mean",
        "missed_margin_mean",
        "waste_cost_mean",
        "episodes",
    ]

    display_cols_err = [
        "policy",
        "final_cum_profit_std",
        "final_cum_profit_95ci",
        "profit_se",
        "profit_moe",
        "profit_ci_low",
        "profit_ci_high"
    ]

    print("\nPolicy Performance — Means Across Episodes")
    print_boxed_table(summary[display_cols])
    print(f"\nSaved per-day results to: {out_path}")

    print("\nError Metrics for Final Cumulative Profit (SE, MOE, 95% CI)")
    print_boxed_table(summary[display_cols_err])
    print(f"\nSaved per-day results to: {out_path}")

# ---- small utils ----
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v

if __name__ == "__main__":
    main()
