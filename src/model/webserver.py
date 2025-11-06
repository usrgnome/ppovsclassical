# server.py
import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Optional
import random
from statistics import NormalDist

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

import hashlib
import math

# ---- your env imports ----
from rl_inventory_q import InventoryPricingEnv
# from stable_baselines3 import DQN  # optional


# ---- small utils ----
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("retro-mart/ws")

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

def expected_visitors_today(g: InventoryPricingEnv, price: float) -> int:
    base = int(g._visitors_for_day(g.day))  # deterministic precomputed visitors
    mult = _traffic_multiplier_from_price(g.traffic_cfg, price, g.config["price_range"])
    return max(0, int(round(base * mult)))

def expected_demand_today(g: InventoryPricingEnv, price: float) -> float:
    conv = _conv_prob(g.conv_cfg, g.config["price_range"], price)
    return float(expected_visitors_today(g, price) * conv)

class NewsvendorPolicy():
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

    def _demand_stats(self, g: InventoryPricingEnv, p: float) -> tuple[float, float]:
        mu1 = float(expected_demand_today(g, p))
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

    def act(self, obs: np.ndarray, g: InventoryPricingEnv) -> np.ndarray:
        lo, hi = g.config.get("price_range")[0], g.config.get("price_range")[1]
        cap, max_restock = g.config.get("inventory_capacity"), g.config.get("max_restock")

        opening_inv = int(g.inventory)
        cap_left = max(0, cap - opening_inv)

        restock_cost = float(g.config["restock_cost"])
        holding_cost = float(g.config.get("holding_cost", 0.0))
        stockout_penalty = 0.0

        n_steps = max(2, int(round((hi - lo) / self.price_step)) + 1)
        prices = np.linspace(lo, hi, n_steps)

        best_profit = -1e18
        best_price = lo
        best_restock_units = 0

        for p in prices:
            mu, var = self._demand_stats(g, float(p))

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

globalPolicy = NewsvendorPolicy()


def _canonicalize(obj, precision=8):
    """Convert to a JSON-serializable, deterministically ordered structure."""
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k], precision) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v, precision) for v in obj]
    if isinstance(obj, float):
        if math.isfinite(obj):
            x = round(obj, precision)
            return 0.0 if x == 0 else x
        return str(obj)
    return obj

def hash_config(cfg, precision=8):
    canon = _canonicalize(cfg, precision=precision)
    payload = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# ---------------- JSON helpers ----------------
def to_jsonable(obj: Any) -> Any:
    if dataclass_isinstance(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj

def dataclass_isinstance(x: Any) -> bool:
    return hasattr(x, "__dataclass_fields__")

def dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), separators=(",", ":"))

# ---------------- Data contracts ----------------
API_VERSION = "1.0.0"

@dataclass
class DayEntry:
    day: int
    price: float
    sales: int
    revenue: float
    profit: float
    episodeProfit: float
    openingStock: int
    restocked: int
    expired: int
    footTraffic: int
    holding_cost: float
    restock_cost: float
    # Derived KPIs
    conversion: float
    aov: float
    sellThrough: float
    waste: int

@dataclass
class Snapshot:
    step: List[float]
    day: int
    totalDays: int
    price: float
    done: bool
    batches: List[Tuple[int, int]]
    expireDays: int
    inventoryLeft: int
    inventoryCapacity: int
    aiPrice: float
    aiRestock: int

@dataclass
class InitPayload:
    snapshot: Snapshot
    history: List[DayEntry]
    footTrafficSeries: List[int]
    message: str

@dataclass
class StepPayload:
    snapshot: Snapshot
    history: List[DayEntry]

# ---------------- Per-connection state ----------------
class ClientState:
    def __init__(self) -> None:
        self.env: Optional[InventoryPricingEnv] = None
        self.history: List[DayEntry] = []
        self.seed: Optional[int] = None
        # self.model = DQN.load("dqn_inventory_final")

clients: Dict[WebSocketServerProtocol, ClientState] = {}

# ---------------- KPI helpers ----------------
def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v

def compute_kpis(
    *,
    sales: int,
    revenue: float,
    price: float,
    opening: int,
    restocked: int,
    expired: int,
    foot_traffic: int,
) -> Tuple[float, float, float, int]:
    conversion = (sales / foot_traffic) if foot_traffic > 0 else 0.0
    aov = (revenue / sales) if sales > 0 else price
    denom = max(1, opening + restocked)
    sell_through = sales / denom
    waste = expired
    return conversion, aov, sell_through, waste

def extract_env_numbers(env: InventoryPricingEnv) -> Dict[str, Any]:
    """Pulls commonly needed numbers with defensive fallbacks."""
    return {
        "day": safe_int(getattr(env, "day", 1)),
        "total_days": safe_int(env.config.get("max_days", 1)),
        "price": safe_float(getattr(env, "price", 0.0)),
        "last_sales": safe_int(getattr(env, "last_sales", 0)),
        "last_revenue": safe_float(getattr(env, "last_revenue", 0.0)),
        "last_profit": safe_float(getattr(env, "last_profit", 0.0)),
        "episode_profit": safe_float(getattr(env, "profit", 0.0)),
        "last_foot_traffic": safe_int(getattr(env, "last_foot_traffic", 0)),
        "opening_stock": safe_int(getattr(env, "last_opening_inventory", 0)),
        "restock_cost": safe_int(getattr(env, "last_restock_cost", 0)),
        "holding_cost": safe_int(getattr(env, "last_holding_cost", 0)),
        "restocked": safe_int(getattr(env, "last_restocked", 0)),
        "expired": safe_int(getattr(env, "last_expired", 0)),
        "inventory_left": safe_int(getattr(env, "inventory", 0)),
        "expire_days": safe_int(env.config.get("expiration_days", 0)),
        "batches": [(safe_int(d), safe_int(q)) for d, q in getattr(env, "batches", [])],
        "foot_series": [safe_int(x) for x in getattr(env, "foot_traffic", [])],
        "inventory_capacity": safe_int(env.config.get("inventory_capacity", 0)),
    }

def make_day_entry(env: InventoryPricingEnv) -> DayEntry:
    e = extract_env_numbers(env)
    conversion, aov, sell_through, waste = compute_kpis(
        sales=e["last_sales"],
        revenue=e["last_revenue"],
        price=e["price"],
        opening=e["opening_stock"],
        restocked=e["restocked"],
        expired=e["expired"],
        foot_traffic=e["last_foot_traffic"],
    )
    return DayEntry(
        day=max(0, e["day"] - 1),
        price=e["price"],
        sales=e["last_sales"],
        holding_cost=e["holding_cost"],
        restock_cost=e["restock_cost"],
        revenue=e["last_revenue"],
        profit=e["last_profit"],
        episodeProfit=e["episode_profit"],
        openingStock=e["opening_stock"],
        restocked=e["restocked"],
        expired=e["expired"],
        footTraffic=e["last_foot_traffic"],
        conversion=conversion,
        aov=aov,
        sellThrough=sell_through,
        waste=waste,
    )

def make_snapshot(env: InventoryPricingEnv, state_vec: np.ndarray, terminated: bool, aiPrice: float, aiRestock: int) -> Snapshot:
    e = extract_env_numbers(env)
    return Snapshot(
        step=np.asarray(state_vec, dtype=float).tolist(),
        day=max(0, e["day"] - 1),
        totalDays=e["total_days"],
        price=e["price"],
        done=bool(terminated),
        batches=e["batches"],
        expireDays=e["expire_days"],
        inventoryLeft=e["inventory_left"],
        inventoryCapacity=e["inventory_capacity"],
        aiPrice=aiPrice,
        aiRestock=aiRestock,
    )

# ---------------- Config plumbing ----------------

DEFAULT_TRAFFIC_MULT = [0.90, 0.85, 1.00, 1.00, 1.10, 1.20, 1.30]

def _sanitize_weekday_mult(arr: Any) -> List[float]:
    try:
        xs = list(arr)
    except Exception:
        return DEFAULT_TRAFFIC_MULT[:]
    xs = [safe_float(v, 1.0) for v in xs][:7]
    if len(xs) < 7:
        xs += [1.0] * (7 - len(xs))
    return xs

def _build_demand_config(body: Dict[str, Any]) -> Dict[str, Any]:
    """Map UI payload.demandConfig into env.demand_config, with sane defaults."""
    dc = body.get("demandConfig", {}) or {}

    # ---- FOOT TRAFFIC ----
    traffic = dc.get("traffic", {})
    # Back-compat: allow flat fields (baseDemand, weekdayMultipliers, normalStdDev)
    base = safe_int(traffic.get("base", dc.get("baseDemand", body.get("baseDemand", 50))))
    weekday_mult = _sanitize_weekday_mult(
        traffic.get("weekdayMultipliers", dc.get("weekdayMultipliers", DEFAULT_TRAFFIC_MULT))
    )
    weekday_start = (traffic.get("weekdayStart") or "Mon")
    visitor_noise_std = safe_float(traffic.get("visitorNoiseStd", max(3.0, base * 0.12)))
    price_traffic_cut = bool(traffic.get("priceTrafficCut", False))
    price_traffic_delta = safe_float(traffic.get("priceTrafficDelta", body.get("maxPrice", 20.0)))

    traffic_cfg = {
        "base": base,
        "weekdayMultipliers": weekday_mult,
        "weekdayStart": "Sun" if weekday_start == "Sun" else "Mon",
        "visitorNoiseStd": visitor_noise_std,
        "priceTrafficCut": price_traffic_cut,
        "priceTrafficDelta": price_traffic_delta,
    }

    # ---- CONVERSION ----
    conv = dc.get("conversion", {})
    # Back-compat: demandFunc/linear/exp params
    model = (conv.get("model") or dc.get("demandFunc") or "wtpLogNormal")

    p1 = safe_float(conv.get("p1", 5.0))
    c1 = clamp01(conv.get("c1", 0.40))
    p2 = safe_float(conv.get("p2", 15.0))
    c2 = clamp01(conv.get("c2", 0.10))

    p0 = safe_float(conv.get("p0", 10.0))
    c0 = clamp01(conv.get("c0", 0.20))
    elasticity = safe_float(conv.get("elasticity", -1.5))

    linear_k = safe_float(conv.get("linearPriceCoeff", dc.get("linearPriceCoeff", 1.0)))
    exp_alpha = safe_float(conv.get("expElasticity", dc.get("expElasticity", 0.05)))

    conversion_cfg = {
        "model": str(model),
        "p1": p1, "c1": c1, "p2": p2, "c2": c2,
        "p0": p0, "c0": c0, "elasticity": elasticity,
        "linearPriceCoeff": linear_k,
        "expElasticity": exp_alpha,
    }

    # ---- DEMAND NOISE ----
    noise = dc.get("noise", {})
    noise_model = (noise.get("model") or "poisson")
    negbin_k = safe_float(noise.get("negbinK", 5.0))
    noise_cfg = {"model": str(noise_model), "negbinK": negbin_k}

    return {"traffic": traffic_cfg, "conversion": conversion_cfg, "noise": noise_cfg}

def _build_env_config(body: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the InventoryPricingEnv config dict from incoming init payload."""
    max_days = safe_int(body.get("simulationDays", 7))
    initial_inventory = safe_int(body.get("initialInventory", 100))
    shelf_life = safe_int(body.get("shelfLifeDays", 7))
    restock_cost = safe_float(body.get("restockCost", 4.0))
    holding_cost = safe_float(body.get("holdingCost", 1.0))
    base_demand_legacy = safe_float(body.get("baseDemand", 50))  # used only as a fallback
    price_hi = safe_float(body.get("maxPrice", 20.0))
    init_price = safe_float(body.get("initialPrice", 10.0))
    inventory_capacity = safe_int(body.get("inventoryCapacity", 200))
    max_restock = safe_int(body.get("maxRestock", 100))  # NEW: accept override; fallback 100

    env_cfg = {
        "max_days": max_days,
        "initial_inventory": initial_inventory,
        "expiration_days": shelf_life,
        "restock_cost": restock_cost,
        "holding_cost": holding_cost,
        "base_demand": base_demand_legacy,       # legacy—env will prefer demand_config.traffic.base
        "price_range": (1.0, price_hi),
        "max_restock": max_restock,
        "inventory_capacity": inventory_capacity,
        "initial_price": init_price,
        # NEW nested demand config
        "demand_config": _build_demand_config(body),
    }
    return env_cfg

# ---------------- Message helpers ----------------
async def send_json(ws: WebSocketServerProtocol, msg_type: str, payload: Any, meta: Optional[Dict[str, Any]] = None) -> None:
    meta = {"version": API_VERSION, **(meta or {})}
    await ws.send(dumps({"type": msg_type, "payload": payload, "meta": meta}))

async def send_error(ws: WebSocketServerProtocol, message: str, code: str = "bad_request") -> None:
    await send_json(ws, "error", {"message": message, "code": code})

# ---------------- Handler ----------------
async def handler(ws: WebSocketServerProtocol) -> None:
    clients[ws] = ClientState()
    log.info("client connected %s", ws.remote_address)
    await send_json(ws, "handshake", {"ok": True}, meta={"state": 0})

    try:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await send_error(ws, "Invalid JSON")
                continue

            cmd = data.get("type")
            body = data.get("payload", {}) or data  # allow legacy shape
            state = clients[ws]

            if cmd == "init":
                # Build env config (supports new nested demandConfig)
                env_cfg = _build_env_config(body)
                env = InventoryPricingEnv(env_cfg)

                log.info("Initialized env cfg hash=%s", hash_config(env_cfg))
                log.debug("Env config: %s", env.config)
                print(env_cfg)

                state.env = env
                state.history.clear()


                # Prime day 0: take a no-opish step so UI has metrics for day 0
                obs, reward, terminated, truncated, _ = env._step(env.config.get("initial_price", 1.0), 0)
                state.history.append(make_day_entry(env))
                
                res = globalPolicy.act(obs, env)
                res_price = float(res[0])
                res_restock = int(round(res[1] * env.config.get("max_restock", 0)))
                print("AI recommends price", res)

                snap = make_snapshot(env, obs, terminated, res_price, res_restock)
                payload = InitPayload(
                    snapshot=snap,
                    history=state.history[:],
                    footTrafficSeries=extract_env_numbers(env)["foot_series"],
                    message="Environment initialized",
                )
                await send_json(ws, "init", payload)

            elif cmd == "reset":
                if not state.env:
                    await send_error(ws, "Environment not initialized", code="not_initialized")
                    continue
                env = state.env
                env.reset()
                state.history.clear()
                # neutral step for day 0 metrics
                obs, reward, terminated, truncated, _ = env._step(getattr(env, "price", 0.0), 0)
                state.history.append(make_day_entry(env))
                await send_json(ws, "reset", {
                    "snapshot": make_snapshot(env, obs, terminated, 1.0, 1),
                    "history": state.history[:],
                    "footTrafficSeries": extract_env_numbers(env)["foot_series"],
                })

            elif cmd == "step":
                if not state.env:
                    await send_error(ws, "Environment not initialized", code="not_initialized")
                    continue

                env = state.env
                price = safe_float(body.get("price", getattr(env, "price", 0.0)))
                restock = safe_int(body.get("restock", 0))

                obs, reward, terminated, truncated, _ = env._step(price, restock)
                env.render(mode="human")



                state.history.append(make_day_entry(env))

                res = globalPolicy.act(obs, env)
                res_price = float(res[0])
                res_restock = int(round(res[1] * env.config.get("max_restock", 0)))
                print("AI recommends price", res)
                
                payload = StepPayload(
                    snapshot=make_snapshot(env, obs, terminated, res_price, res_restock),
                    history=state.history[:],
                )
                await send_json(ws, "step", payload)

            elif cmd == "render":
                if not state.env:
                    await send_error(ws, "Environment not initialized", code="not_initialized")
                    continue
                state.env.render("human")
                await send_json(ws, "render", {"ok": True})

            else:
                await send_error(ws, f"Unknown command: {cmd}")

    except websockets.exceptions.ConnectionClosed:
        log.info("client disconnected %s", ws.remote_address)
    except Exception as e:
        log.exception("unhandled server error")
    finally:
        clients.pop(ws, None)

# ---------------- Server bootstrap ----------------
async def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    log.info("starting WebSocket: ws://%s:%d", host, port)
    async with websockets.serve(
        handler,
        host,
        port,
        ping_interval=20,
        ping_timeout=20,
        max_size=2_000_000,
        max_queue=64,
        compression="deflate",
    ):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("server stopped")
