import numpy as np
import gymnasium as gym
from gymnasium import spaces
from rl_inventory_q import InventoryPricingEnv
from collections import deque

class InventoryPPOEnv(gym.Env):
    """
    Actions: [ price_abs, restock_fraction ]
      price_abs ∈ [price_lo, price_hi]
      restock_fraction ∈ [0,1] of today's fillable space (capped by max_restock)

    Observations (all in [0,1]; dynamically sized):
      Core:
        [inv_frac, day_frac, dow_sin, dow_cos,
         price_frac, last_price_frac, price_change_frac,
         ft_today_frac, ft_tmrw_frac, cap_left_frac,
         unmet_frac, last_conv]
      Age buckets (size = expiration_days):
         [qty_expiring_in_1d .. qty_expiring_in_Kd] / capacity
      Recent flows:
         [last_sales_frac, last_expired_frac, last_restocked_frac]
      Config scalars (normalized):
         [restock_cost_frac, holding_cost_frac, waste_cost_frac,
          max_restock_over_cap, expiration_days_over_10]
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, base_config=None, seed=None,
                 price_jump_penalty: float = 0.0,
                 stockout_penalty: float = 0.0,
                 render_mode: str | None = None):
        super().__init__()
        self.env = InventoryPricingEnv(config=base_config or {}, seed=seed)

        # penalties
        self._price_jump_penalty = float(price_jump_penalty)
        self._stockout_penalty   = float(stockout_penalty)

        # cache config
        self._price_lo, self._price_hi = map(float, self.env.config["price_range"])
        self._cap         = int(self.env.config["inventory_capacity"])
        self._max_days    = int(self.env.config["max_days"])
        self._max_restock = int(self.env.config["max_restock"])
        self._exp_days    = int(self.env.config["expiration_days"])

        # rendering
        self.render_mode = render_mode or "human"

        # trackers
        self._last_price = float(self.env.price)
        self._last_price_frac = 0.0
        self._last_price_change_frac = 0.0
        self._vis_cap    = 1.0  # fixed per-episode cap computed in reset()

        self.real_profit = 0.0

        # Build a first obs to lock obs space
        first_obs = self._make_obs(initialize=True)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=first_obs.shape, dtype=np.float32
        )

        # actions: price absolute, restock fraction
        self.action_space = spaces.Box(
            low=np.array([self._price_lo, 0.0], dtype=np.float32),
            high=np.array([self._price_hi, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    # ---------- Gym API ----------
    def reset(self, seed: int | None = None, options=None):
        _, _ = self.env.reset(seed=seed)
        self._last_price = float(self.env.price)
        self._last_price_frac = self._price_to_frac(self._last_price)
        self._last_price_change_frac = 0.0

        # Fix traffic cap once from base env's foot_traffic list
        ft = getattr(self.env, "foot_traffic", None)
        if not ft:
            raise Exception("No foot traffic")
        self._vis_cap = max(1.0, float(max(ft))) if ft and len(ft) > 0 else 1.0

        first_obs = self._make_obs()
        return first_obs, {}

    def step(self, action):
        lo, hi = self._price_lo, self._price_hi

        # parse & clip actions
        price = float(np.clip(action[0], lo, hi))
        frac  = float(np.clip(action[1], 0.0, 1.0))

        # price jump vs previous price (compute BEFORE stepping)
        price_jump = abs(price - self._last_price) / max(1e-6, (hi - lo))

        raw_order = int(frac * self._max_restock)
        cap_left  = max(0, self._cap - int(self.env.inventory))
        restock   = min(raw_order, cap_left)

        # step base env
        _, _, terminated, truncated, info = self.env._step(price, restock)
        self.real_profit = float(self.env.profit)

        # base reward
        reward = float(self.env.last_profit)

        # shaping: price jump
        if self._price_jump_penalty > 0.0:
            reward -= self._price_jump_penalty * np.clip(price_jump, 0.0, 1.0)

        # shaping: stockouts
        if self._stockout_penalty > 0.0:
            unmet = float(getattr(self.env, "last_unmet_demand", 0.0))
            if unmet > 0.0:
                # prefer unmet * (positive margin) but keep it simple/stable
                penalty = self._stockout_penalty * unmet
                reward -= penalty

        # small margin encouragement (optional)
        unit_margin = max(0.0, float(self.env.price) - float(self.env.config["restock_cost"]))
        reward += 0.5 * unit_margin

        # ---- update trackers for next observation ----
        # compute price change in [0,1] space and store
        curr_price_frac = self._price_to_frac(float(self.env.price))
        self._last_price_change_frac = np.clip(abs(curr_price_frac - self._last_price_frac), 0.0, 1.0)
        self._last_price_frac = curr_price_frac
        self._last_price = float(self.env.price)

        if self.render_mode == "human":
            self.env.render(self.render_mode)

        return self._make_obs(), reward, bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render()

    # ---------- Helpers ----------
    def _price_to_frac(self, p: float) -> float:
        return float(np.clip((p - self._price_lo) / max(1e-6, (self._price_hi - self._price_lo)), 0.0, 1.0))

    # ---------- Observation builder ----------
    def _make_obs(self, initialize: bool=False) -> np.ndarray:
        inv = float(self.env.inventory)
        cap = float(self._cap)
        inv_frac = np.clip(inv / max(1.0, cap), 0.0, 1.0)

        # day fraction & weekday
        day_frac = np.clip(float(self.env.day) / max(1.0, float(self._max_days)), 0.0, 1.0)
        dow      = int(self.env.day) % 7
        dow_sin  = 0.5 * (np.sin(2.0 * np.pi * dow / 7.0) + 1.0)
        dow_cos  = 0.5 * (np.cos(2.0 * np.pi * dow / 7.0) + 1.0)

        # prices
        price_frac = self._price_to_frac(float(self.env.price))
        last_price_frac = self._last_price_frac if not initialize else price_frac
        price_change_frac = self._last_price_change_frac if not initialize else 0.0

        # foot traffic today & tomorrow
        vden = max(1.0, self._vis_cap)
        ft_today_frac = np.clip(float(self.env._visitors_for_day(self.env.day))     / vden, 0.0, 1.0)
        ft_tmrw_frac  = np.clip(float(self.env._visitors_for_day(self.env.day + 1)) / vden, 0.0, 1.0)

        # capacity left
        cap_left_frac = float(max(0.0, (self.env.config["inventory_capacity"] - self.env.inventory)
                                  / max(1.0, self.env.config["inventory_capacity"])))

        # unmet fraction (yesterday’s unmet / yesterday’s visitors)
        if self.env.day == 0:
            unmet_frac = 0.0
        else:
            y_vis = max(1.0, float(self.env._visitors_for_day(self.env.day - 1)))
            unmet_frac = np.clip(float(getattr(self.env, "last_unmet_demand", 0.0)) / y_vis, 0.0, 1.0)

        # last conversion
        last_conv = float(np.clip(getattr(self.env, "last_conversion_rate", 0.0), 0.0, 1.0))

        # ---------- NEW: inventory age buckets ----------
        exp_days = self._exp_days
        age_buckets = [0.0] * exp_days  # index 0 = expires tomorrow
        for d, q in self.env.batches:
            age = self.env.day - d  # 0 = fresh
            days_to_expire = exp_days - age
            idx = max(0, min(exp_days - 1, days_to_expire - 1))
            age_buckets[idx] += float(q)
        age_frac = (np.array(age_buckets, dtype=np.float32) / max(1.0, cap)).astype(np.float32)

        # ---------- NEW: recent flows (normalized by capacity) ----------
        last_sales_frac     = np.clip(float(getattr(self.env, "last_sales", 0.0))     / max(1.0, cap), 0.0, 1.0)
        last_expired_frac   = np.clip(float(getattr(self.env, "last_expired", 0.0))   / max(1.0, cap), 0.0, 1.0)
        last_restocked_frac = np.clip(float(getattr(self.env, "last_restocked", 0.0)) / max(1.0, cap), 0.0, 1.0)

        # ---------- NEW: config scalars, normalized ----------
        price_scale = max(1e-6, self._price_hi)  # normalize costs by max price
        restock_cost_frac  = np.clip(float(self.env.config["restock_cost"]) / price_scale, 0.0, 1.0)
        holding_cost_frac  = np.clip(float(self.env.config["holding_cost"]) / price_scale, 0.0, 1.0)
        waste_cost_frac    = np.clip(float(self.env.config["waste_cost"])   / price_scale, 0.0, 1.0)
        max_restock_over_cap = np.clip(float(self._max_restock) / max(1.0, cap), 0.0, 1.0)
        expiration_days_over_10 = np.clip(float(self._exp_days) / 10.0, 0.0, 1.0)

        core = np.array([
            inv_frac,
            day_frac,
            dow_sin, dow_cos,
            price_frac, last_price_frac, price_change_frac,
            ft_today_frac, ft_tmrw_frac,
            cap_left_frac,
            unmet_frac,
            last_conv,
        ], dtype=np.float32)

        flows = np.array([
            last_sales_frac, last_expired_frac, last_restocked_frac
        ], dtype=np.float32)

        cfgs = np.array([
            restock_cost_frac, holding_cost_frac, waste_cost_frac,
            max_restock_over_cap, expiration_days_over_10
        ], dtype=np.float32)

        obs = np.concatenate([core, age_frac, flows, cfgs]).astype(np.float32)
        return obs

class InventoryPPOEnvStack(gym.Wrapper):
    """
    Stacks the last K 1D observations along the last dimension.
    At reset, repeats the first observation K times (or zero-pads if zero_pad=True).
    """
    def __init__(self, env: gym.Env, k: int, zero_pad: bool = False, dtype=np.float32):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) == 1, \
            "FrameStack1D expects a 1D Box observation space"

        self.k = int(k)
        self.zero_pad = bool(zero_pad)
        self.dtype = dtype

        self.obs_dim = env.observation_space.shape[0]
        low  = np.repeat(env.observation_space.low,  self.k, axis=0)
        high = np.repeat(env.observation_space.high, self.k, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=dtype)

        self._frames = deque(maxlen=self.k)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        if self.zero_pad:
            for _ in range(self.k - 1):
                self._frames.append(np.zeros_like(obs, dtype=self.dtype))
            self._frames.append(obs.astype(self.dtype))
        else:
            for _ in range(self.k):
                self._frames.append(obs.astype(self.dtype))
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs.astype(self.dtype))
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        return np.concatenate(list(self._frames), axis=0).astype(self.dtype)