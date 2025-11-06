import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ql_wrapper import InventoryPPOEnv  # your wrapper

# Keep this in sync with training
BASE_CONFIG = dict(
    max_days=30,
    initial_inventory=30,
    expiration_days=7,
    restock_cost=1.2,
    holding_cost=0.01 * 0,          # NEW
    base_demand=50,
    price_range=(1.0, 6.0),
    max_restock=150,
    inventory_capacity=220,
    initial_price=3.5,
    demand_config=dict(
        traffic=dict(
            base=50,
            weekdayMultipliers=[0.9,0.9,1.0,1.0,1.1,1.3,1.4],
            weekdayStart="Mon",
            visitorNoiseStd=40.0,
            priceTrafficCut=False,
            priceTrafficDelta=15.0,
            # visitorNoiseTruncStd=1.0,
        ),
        conversion=dict(
            model="wtpLogNormal",
            p1=3.0, c1=0.10,
            p2=5.0, c2=0.04,
            p0=10.0, c0=0.20, elasticity=-0.5,
            linearPriceCoeff=1.0,
            expElasticity=0.05,
        ),
        noise=dict(
            model="poisson",
            negbinK=5.0,
        ),
    ),
)


def ppo_policy_from_model(model):
    def pf(env):
        obs = []  # you have empty state; if you later add state, pass it here
        action, _ = model.predict(obs, deterministic=True)
        price = float(np.clip(action[0], *env.config["price_range"])) if np.ndim(action)>0 else float(action)
        restock = int(round(action[1])) if np.ndim(action)>1 else 0
        return price, max(0, min(restock, env.config["max_restock"]))
    return pf

def fixed_price_policy(price, threshold=40, restock_qty=80):
    def pf(env):
        inv = env.inventory
        r = restock_qty if inv < threshold else 0
        return price, r
    return pf

def random_policy(env):
    lo, hi = env.config["price_range"]
    return env._rng.uniform(lo, hi), env._rng.randint(0, env.config["max_restock"])

def make_env(seed_offset=0, price_jump_penalty=0.05, stockout_penalty=0.1):
    def _thunk():
        seed = np.random.randint(1_000_000) + seed_offset
        return InventoryPPOEnv(
            base_config=BASE_CONFIG,
            seed=12345,
            price_jump_penalty=0,
            stockout_penalty=0.01,
            render_mode="human"
        )
    return _thunk


def run_one_episode(model) -> float:
    env_fn = make_env(999)              # this is the thunk (callable)
    venv = DummyVecEnv([env_fn])        # pass the thunk directly
    inner_env: InventoryPPOEnv = venv.envs[0]

    obs = venv.reset()
    ep_profit = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print("Obs:", obs.tolist())
        print("Action:", action.tolist())
        obs, rewards, dones, infos = venv.step(action)
        ep_profit += float(rewards[0])
        done = bool(dones[0])
    return inner_env.real_profit

if __name__ == "__main__":
    model = PPO.load("./ppo_inventory_policy")
    profits = [run_one_episode(model) for _ in range(1)]
    print("Episode profits:", [f"{p:.2f}" for p in profits])
    print(f"Average final profit: {np.mean(profits):.2f}")