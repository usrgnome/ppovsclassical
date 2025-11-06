from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import math
import random

Number = float | int


@dataclass
class EnvConfig:
    max_days: int = 30
    initial_inventory: int = 50
    holding_cost: float = 0.0        # per-unit per-day holding cost
    waste_cost: float = 0.0        # per-unit per-day holding cost
    expiration_days: int = 5
    restock_cost: float = 2.0
    base_demand: int = 50            # legacy baseline daily visitors (used if demand_config.traffic.base absent)
    price_range: Tuple[float, float] = (1.0, 100.0)
    max_restock: int = 100
    inventory_capacity: int = 200
    initial_price: float = 0      # default opening price


class InventoryPricingEnv:
    """
    Inventory + pricing simulator with configurable foot traffic, conversion, and demand noise.

    New nested config (optional):
      config["demand_config"] = {
        "traffic": {
          "base": int,
          "weekdayMultipliers": [float]*7,   # UI may be Sun..Sat; rotate with weekdayStart
          "weekdayStart": "Mon"|"Sun",
          "visitorNoiseStd": float,          # std dev of visitor noise, absolute units
          "priceTrafficCut": bool,           # if True: high prices reduce visitors
          "priceTrafficDelta": float,        # pivot price for traffic suppression
        },
        "conversion": {
          "model": "wtpLogNormal"|"logitLogPrice"|"elasticity"|"linear"|"exponential",
          # two-point calibration (used by wtp/logit)
          "p1": float, "c1": float, "p2": float, "c2": float,
          # elasticity model
          "p0": float, "c0": float, "elasticity": float,  # elasticity < 0
          # legacy simple models
          "linearPriceCoeff": float,
          "expElasticity": float,
        },
        "noise": {
          "model": "poisson"|"binomial"|"negbin",
          "negbinK": float,                 # overdispersion param (>0)
        },
      }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        config = dict(config or {})

        # keep supported top-level keys
        allowed_keys = EnvConfig.__dataclass_fields__.keys()
        clean_config = {k: v for k, v in config.items() if k in allowed_keys}
        cfg = EnvConfig(**clean_config)

        # normalize/validate core fields
        self.config: Dict[str, Number] = dict(
            max_days=int(cfg.max_days),
            initial_inventory=int(cfg.initial_inventory),
            expiration_days=int(cfg.expiration_days),
            restock_cost=float(cfg.restock_cost),
            base_demand=int(cfg.base_demand),
            price_range=(float(cfg.price_range[0]), float(cfg.price_range[1])),
            max_restock=int(cfg.max_restock),
            inventory_capacity=int(cfg.inventory_capacity),
            initial_price=float(cfg.initial_price),
            holding_cost=float(cfg.holding_cost),
            waste_cost=float(cfg.waste_cost),
        )

        # NEW: nested demand config (traffic, conversion, noise)
        self.demand_cfg: Dict[str, Any] = config.get("demand_config", {}) or {}
        self.traffic_cfg: Dict[str, Any] = self.demand_cfg.get("traffic", {}) or {}
        self.conv_cfg: Dict[str, Any] = self.demand_cfg.get("conversion", {}) or {}
        self.noise_cfg: Dict[str, Any] = self.demand_cfg.get("noise", {}) or {}

        # RNG
        self._rng = random.Random(seed)

        # public state used by server/UI
        self.day: int = 0
        self.price: float = 0
        self.inventory: int = 0
        self.batches: List[Tuple[int, int]] = []  # list of (day_created, qty)
        self.foot_traffic: List[int] = []

        # last tick metrics (server reads these)
        self.last_profit: float = 0.0
        self.last_sales: int = 0
        self.last_holding_cost: float = 0.0
        self.last_restock_cost: float = 0.0
        self.last_revenue: float = 0.0
        self.last_foot_traffic: int = 0
        self.last_expired: int = 0
        self.last_opening_inventory: int = 0
        self.last_restocked: int = 0
        self.last_unmet_demand: int = 0

        # running totals
        self.profit: float = 0.0

        # Initialize
        self.reset(seed=seed)

    # --------- Public API ---------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng.seed(seed)

        self.day = 0
        self.price = 0
        self.batches = [(self.day, int(self.config["initial_inventory"]))]
        self.inventory = int(self.config["initial_inventory"])
        self.profit = 0.0

        self.last_profit = 0.0
        self.last_sales = 0
        self.last_revenue = 0.0
        self.last_holding_cost = 0.0
        self.last_restock_cost = 0.0
        self.last_foot_traffic = 0
        self.last_unmet_demand = 0
        self.last_conversion_rate = 0
        self.last_expired = 0
        self.last_opening_inventory = self.inventory
        self.last_restocked = 0

        self.foot_traffic = self._generate_foot_traffic()

        state_vec = self._create_state(demand=0, sales=0)
        info: Dict = {}
        return state_vec, info

    def _step(self, price: float, restock_qty: int):
        """
        Progress one day with player's chosen price and restock.
        Returns (state_vec, reward=0.0, terminated, truncated, info)
        """
        # Clamp inputs
        lo, hi = self.config["price_range"]
        price = float(_clamp(price, lo, hi))
        restock_qty = int(max(0, min(restock_qty, self.config["max_restock"])))

        self.price = price

        # Remove expired inventory at start of day
        expired = self._remove_expired()
        self.last_expired = expired

        # Opening inventory after expiry (before restock)
        self.last_opening_inventory = self.inventory

        # Restock (respect capacity)
        capacity_left = int(self.config["inventory_capacity"]) - self.inventory
        actual_restock = min(restock_qty, max(capacity_left, 0))
        if actual_restock > 0:
            self.batches.append((self.day, actual_restock))
            self.inventory += actual_restock
        self.last_restocked = actual_restock

        # Today's visitors (precomputed) and demand
        today_visitors = self._visitors_for_day(self.day)
        self.last_foot_traffic = today_visitors 

        demand = self._draw_demand(price, today_visitors)

        # Fulfill demand from inventory (FIFO)
        sales = self._fulfill_sales(demand)

        # Economics
        revenue = sales * price
        restock_cost = actual_restock * float(self.config["restock_cost"])
        holding_cost = self.inventory * float(self.config["holding_cost"])
        waste_cost = expired * float(self.config["waste_cost"])
        
        day_profit = revenue - restock_cost - holding_cost - waste_cost

        self.profit += day_profit

        # Update "last" metrics (for server)
        self.last_sales = sales
        self.last_revenue = revenue
        self.last_profit = day_profit
        self.last_restock_cost = restock_cost
        self.last_unmet_demand = demand - sales
        self.last_holding_cost = holding_cost

        # Advance day
        self.day += 1

        # Prepare return values
        state_vec = self._create_state(demand=demand, sales=sales)
        terminated = self.day >= int(self.config["max_days"])
        truncated = False
        info: Dict = {}

        # reward is unused in the game version (keep 0.0 to maintain tuple shape)
        return state_vec, 0, terminated, truncated, info

    def render(self, mode: str = "human"):
        print(
            f"Day: {self.day} / {int(self.config['max_days'])}, "
            f"Yesterdays visitors: {self.last_foot_traffic}, "
            f"Inventory: {self.inventory}, "
            f"Yesterdays Price: {self.price:.2f}, "
            f"Yesterdays Sales: {self.last_sales}, "
            f"Yesterdays Profit (day): {self.last_profit:.2f}, "
            f"Profit (total): {self.profit:.2f}"
        )

    # --------- Internals: Foot Traffic ---------

    def _generate_foot_traffic(self) -> List[int]:
        """
        Deterministic per-day visitors for the whole run (seeded).
        Includes weekday seasonality and noise.

        Uses demand_config.traffic if provided; otherwise falls back to legacy fields.
        """
        max_days = int(self.config["max_days"])

        # traffic base
        base = int(self.traffic_cfg.get("base", self.config["base_demand"]))

        # weekday multipliers: default Mon..Sun
        default_mult = [0.90, 0.85, 1.00, 1.00, 1.10, 1.20, 1.30]
        day_mult = list(self.traffic_cfg.get("weekdayMultipliers", default_mult))
        if len(day_mult) != 7:
            day_mult = default_mult

        # visitor noise (abs std); default legacy behavior
        sigma = float(self.traffic_cfg.get("visitorNoiseStd", max(3.0, base * 0.12)))

        traffic: List[int] = []
        for d in range(max_days):
            mult = day_mult[d % 7]
            mean_visitors = base * mult
            noise = self._rng.gauss(0.0, sigma)
            v = max(0, int(round(mean_visitors + noise)))
            traffic.append(v)
        return traffic

    def _visitors_for_day(self, index: int) -> int:
        if not self.foot_traffic:
            return 0

        if index >= len(self.foot_traffic):
            return 0
            
        i = max(0, min(index, len(self.foot_traffic) - 1))
        return int(self.foot_traffic[i])

    def _price_to_visitors_multiplier(self, price: float) -> float:
        """
        Optional: reduce foot traffic at high prices if enabled.
        Starts suppressing after 'priceTrafficDelta'; gentle decay, min at 0.1x.
        """
        if not self.traffic_cfg.get("priceTrafficCut", False):
            return 1.0
        pivot = float(self.traffic_cfg.get("priceTrafficDelta", self.config["price_range"][1]))
        p = float(price)
        if p <= pivot:
            return 1.0
        # simple rational decay: half-traffic near 2*pivot, clamp to [0.1, 1.0]
        ratio = p / max(1e-9, pivot)
        mult = 1.0 / (1.0 + (ratio - 1.0))
        return float(max(0.1, min(1.0, mult)))

    # --------- Internals: Conversion ---------

    def _price_to_conversion(self, price: float) -> float:
        """
        Price -> per-visitor purchase probability.
        Chooses model from demand_config.conversion, with sensible defaults.
        """
        model = str(self.conv_cfg.get("model", "wtpLogNormal"))
        p = max(float(price), 1e-9)

        if model == "wtpLogNormal":
            # Log-normal WTP survival: conv = 1 - Phi((ln p - mu)/sigma)
            from statistics import NormalDist
            nd = NormalDist()
            p1 = float(self.conv_cfg.get("p1", 5.0)); c1 = _clamp01(self.conv_cfg.get("c1", 0.40))
            p2 = float(self.conv_cfg.get("p2", 15.0)); c2 = _clamp01(self.conv_cfg.get("c2", 0.10))
            z1 = nd.inv_cdf(1.0 - c1); z2 = nd.inv_cdf(1.0 - c2)
            denom = (z2 - z1) or 1e-9
            sigma = (math.log(p2) - math.log(p1)) / denom
            mu = math.log(p1) - sigma * z1
            x = (math.log(p) - mu) / sigma
            conv = 1.0 - nd.cdf(x)

        elif model == "logitLogPrice":
            # Logistic in log-price
            def logit(x: float) -> float:
                x = _clamp01(x)
                return math.log(x / (1.0 - x + 1e-12))
            p1 = float(self.conv_cfg.get("p1", 5.0)); c1 = _clamp01(self.conv_cfg.get("c1", 0.40))
            p2 = float(self.conv_cfg.get("p2", 15.0)); c2 = _clamp01(self.conv_cfg.get("c2", 0.10))
            x1, x2 = math.log(p1), math.log(p2)
            L1, L2 = logit(c1), logit(c2)
            beta = -(L2 - L1) / (x2 - x1 + 1e-12)
            alpha = -L1 - beta * x1
            x = math.log(p)
            conv = 1.0 / (1.0 + math.exp(alpha + beta * x))

        elif model == "elasticity":
            # Constant elasticity: conv = c0 * (p/p0)^epsilon
            p0 = float(self.conv_cfg.get("p0", 10.0))
            c0 = _clamp01(self.conv_cfg.get("c0", 0.20))
            eps = float(self.conv_cfg.get("elasticity", -1.5))
            conv = c0 * (p / max(1e-9, p0)) ** eps

        elif model == "linear":
            # Simple linear decay from price_range lo->hi
            lo, hi = self.config["price_range"]
            k = float(self.conv_cfg.get("linearPriceCoeff", 1.0))
            conv = 1.0 - k * ((p - lo) / max(1e-9, (hi - lo)))

        elif model == "exponential":
            # conv = exp(-alpha * p)
            alpha = float(self.conv_cfg.get("expElasticity", 0.05))
            conv = math.exp(-alpha * p)

        else:
            conv = 0.0

        return float(_clamp(conv, 0.0, 1.0))

    # --------- Internals: Demand draw ---------

    def _draw_demand(self, price: float, visitors: int) -> int:
        """
        Draw a stochastic number of would-be purchases (before inventory cap).
        Noise model from demand_config.noise.
        """
        conv = float(_clamp(self._price_to_conversion(price), 0.0, 1.0))
        visitors = max(0, int(visitors))
        lam = visitors * conv
        self.last_conversion_rate = conv

        noise_model = str(self.noise_cfg.get("model", "poisson"))
        if noise_model == "binomial":
            # exact Binomial(n, p) for small/medium n, normal approx for large n
            n = visitors
            p = conv
            if n <= 400:
                # Bernoulli sum
                cnt = 0
                for _ in range(n):
                    if self._rng.random() < p:
                        cnt += 1
                return cnt
            else:
                mean = lam
                var = n * p * (1.0 - p)
                std = math.sqrt(max(1e-9, var))
                from random import gauss
                return max(0, int(round(gauss(mean, std)))), conv

        if noise_model == "negbin":
            # Overdispersed counts: mean=lam, var=lam + lam^2/k
            k = float(self.noise_cfg.get("negbinK", 5.0))
            if lam <= 0 or k <= 0:
                return 0
            rate = _gamma_sample(self._rng, shape=k, scale=lam / k)  # Gamma
            return _poisson_knuth(self._rng, lam=rate)               # then Poisson

        # default: Poisson
        return _poisson_knuth(self._rng, lam=lam)

    # --------- Internals: Inventory mechanics ---------

    def _remove_expired(self) -> int:
        """Remove expired batches; return quantity expired."""
        exp_days = int(self.config["expiration_days"])
        before = sum(q for _, q in self.batches)
        self.batches = [(d, q) for d, q in self.batches if (self.day - d) < exp_days]
        self.inventory = sum(q for _, q in self.batches)
        expired = before - self.inventory
        return int(expired)

    def _fulfill_sales(self, demand: int) -> int:
        """Sell FIFO from batches; return units sold."""
        sales = min(demand, self.inventory)
        remaining = sales
        new_batches: List[Tuple[int, int]] = []
        for d, q in self.batches:
            if remaining <= 0:
                new_batches.append((d, q))
            elif q <= remaining:
                remaining -= q
                # fully consumed batch
            else:
                new_batches.append((d, q - remaining))
                remaining = 0
        self.batches = new_batches
        self.inventory = sum(q for _, q in self.batches)
        return int(sales)

    # ---- State vector (kept for compatibility; UI doesn't need it directly) ----
    def _create_state(self, demand: int, sales: int) -> List[float]:
        return []


# --------- helpers ---------

def _clamp(x: Number, lo: Number, hi: Number) -> Number:
    return lo if x < lo else hi if x > hi else x

def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v

def _poisson_knuth(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    # Knuth's algorithm (good for small/moderate lam)
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1

def _gamma_sample(rng: random.Random, shape: float, scale: float) -> float:
    """
    Gamma(shape=k, scale=θ) sampler:
      - Marsaglia–Tsang for k > 1
      - Ahrens–Dieter for k <= 1
    """
    k = shape
    if k <= 0:
        return 0.0
    if k <= 1.0:
        # Ahrens–Dieter
        while True:
            u = rng.random()
            b = (math.e + k) / math.e
            p = b * u
            if p <= 1.0:
                x = p ** (1.0 / k)
                u2 = rng.random()
                if u2 <= math.exp(-x):
                    return x * scale
            else:
                x = -math.log((b - p) / k)
                u2 = rng.random()
                if u2 <= x ** (k - 1.0):
                    return x * scale
    else:
        # Marsaglia–Tsang
        d = k - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            from random import gauss
            z = gauss(0.0, 1.0)
            v = 1.0 + c * z
            if v <= 0:
                continue
            v = v * v * v
            u = rng.random()
            if u < 1.0 - 0.0331 * (z ** 4):
                return d * v * scale
            if math.log(u) < 0.5 * z * z + d * (1 - v + math.log(v)):
                return d * v * scale
