import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecNormalize
from ql_wrapper import InventoryPPOEnv, InventoryPPOEnvStack
from stable_baselines3.common.monitor import Monitor

BASE_CONFIG = dict(
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


def make_env():
    #return Monitor(InventoryPPOEnv(BASE_CONFIG, stockout_penalty=0, seed=12345, render_mode="bot"))
    K = 3;
    return Monitor(InventoryPPOEnvStack(InventoryPPOEnv(base_config=BASE_CONFIG, seed=0, render_mode="bot"), k=K))


if __name__ == "__main__":

    # vectorize + normalize
    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = PPO("MlpPolicy", venv, verbose=1, n_steps=2048, batch_size=512, n_epochs=10,
    gamma=0.99, gae_lambda=0.95,
    learning_rate=2e-4,
    clip_range=0.3,
    target_kl=0.03,
    ent_coef=0.012,
    vf_coef=0.5, max_grad_norm=0.5)
    model.learn(total_timesteps=2000_000)

    model.save("ppo_inventory_policy.zip")
    venv.save("vecnorm.pkl")  # <-- save normalization stats