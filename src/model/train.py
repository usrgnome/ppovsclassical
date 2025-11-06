import os
import numpy as np

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from ql_wrapper import InventoryPPOEnv

# ----- SKU CONFIG (same as your script) -----
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

# --------- Helpers to build vectorized envs ----------
def make_env(rank: int, base_seed: int = 12345):
    """
    Returns a thunk that creates a single env instance.
    Using separate seeds per worker to decorrelate rollouts.
    """
    def _init():
        return InventoryPPOEnv(
            base_config=BASE_CONFIG,
            seed=base_seed + rank,
            price_jump_penalty=0.0,
            stockout_penalty=1.0,
            render_mode="bot",
        )
    return _init

def make_vec_env(n_envs: int, use_subproc: bool = True):
    if n_envs == 1:
        return DummyVecEnv([make_env(0)])
    return SubprocVecEnv([make_env(i) for i in range(n_envs)]) if use_subproc else DummyVecEnv([make_env(i) for i in range(n_envs)])

if __name__ == "__main__":
    # ------------- Vectorized training env -------------
    n_envs = 8  # tune for your CPU; batch_size must be divisible by n_envs
    env = make_vec_env(n_envs=n_envs, use_subproc=True)
    env = VecMonitor(env) 

    # Separate eval env (no subproc needed)
    eval_env = make_vec_env(n_envs=1, use_subproc=False)

    # ------------- RecurrentPPO model ------------------
    # Notes:
    # - n_steps is per-env, so total rollout length = n_steps * n_envs
    # - batch_size must be divisible by n_envs for RecurrentPPO
    # - lstm_hidden_size controls the LSTM cell size; n_lstm_layers stacks LSTM layers
    policy_kwargs = dict(
        lstm_hidden_size=64,
        n_lstm_layers=1,
        # Optional: net_arch=[dict(pi=[128, 128], vf=[128, 128])],  # MLP around the LSTM
        # ortho_init=True,
        # activation_fn=th.nn.Tanh,  # default is Tanh; keep unless you have a reason to change
    )

    model = RecurrentPPO(
        policy=MlpLstmPolicy,
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,           # per env → 256 * 8 = 2048 rollout length (similar to your PPO)
        batch_size=1024,       # divisible by n_envs (8); typical: 512, 1024, or 2048
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,           # PPO epochs per update
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        # use_sde=False,  # SDE is not supported in RecurrentPPO
        seed=12345,
    )

    # ------------- Callbacks (early stopping / eval) -------------
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path="./rppo_best",
        log_path="./rppo_eval",
        eval_freq=10_000 // n_envs,   # evaluate every ~10k steps
        n_eval_episodes=5,
        deterministic=True,           # important for evaluating recurrent policies
        callback_after_eval=stop_cb,
        render=False,
    )

    # ------------- Train & save ------------------------
    total_timesteps = 1_600_000
    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=True)
    os.makedirs("./models", exist_ok=True)
    model.save("./models/recurrent_ppo_inventory_policy")

    # Optional: also save final eval-best (already saved by EvalCallback)
    print("Training complete. Saved policy to ./models/recurrent_ppo_inventory_policy")
