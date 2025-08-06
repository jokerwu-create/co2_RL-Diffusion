import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from environment import CO2RRMaterialEnv
import torch

# 设置目标分子（乙醇 - 实际CO2RR中更复杂）
TARGET_SMILES = 'CCO'  # 乙醇

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 创建环境
env = DummyVecEnv([lambda: CO2RRMaterialEnv(target_smiles=TARGET_SMILES)])

# 创建PPO模型
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256])
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./logs/",
    device="cuda" if torch.cuda.is_available() else "cpu",
    n_steps=1024,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2
)

# 创建评估回调
eval_env = DummyVecEnv([lambda: CO2RRMaterialEnv(target_smiles=TARGET_SMILES)])
stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_after_eval=stop_callback,
    eval_freq=5000,
    best_model_save_path="./best_model/",
    verbose=1
)

# 训练模型
print("开始训练强化学习智能体...")
model.learn(
    total_timesteps=100000,
    callback=eval_callback,
    tb_log_name="ppo_co2rr"
)

# 保存最终模型
os.makedirs("./models", exist_ok=True)
model.save("./models/ppo_co2rr_final")
print("训练完成! 模型已保存至 ./models/")
