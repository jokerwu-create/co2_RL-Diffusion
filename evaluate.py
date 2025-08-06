import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import CO2RRMaterialEnv
from utils import draw_molecule, plot_molecules
from rdkit.Chem import Descriptors

def evaluate_agent(model_path, num_episodes=10):
    """评估训练好的智能体"""
    # 加载模型
    model = PPO.load(model_path)
    
    # 创建环境
    env = CO2RRMaterialEnv(target_smiles='CCO')
    
    # 运行多个episode
    results = []
    molecules = []
    
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step_rewards = []
        step_molecules = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_rewards.append(reward)
            
            # 记录分子
            if env.current_mol:
                step_molecules.append(env.current_mol)
        
        # 记录结果
        results.append({
            "episode": i+1,
            "total_reward": total_reward,
            "final_smiles": info["smiles"],
            "final_reward": reward,
            "step_rewards": step_rewards,
            "molecules": step_molecules
        })
        molecules.extend(step_molecules)
        
        print(f"Episode {i+1}: Total Reward = {total_reward:.4f}, Final SMILES = {info['smiles']}")
    
    # 可视化结果
    visualize_results(results, molecules)
    
    return results

def visualize_results(results, molecules):
    """可视化评估结果"""
    plt.figure(figsize=(15, 10))
    
    # 1. 奖励曲线
    plt.subplot(2, 2, 1)
    for i, res in enumerate(results):
        plt.plot(res["step_rewards"], label=f"Ep {i+1}", alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward per Step")
    plt.legend()
    
    # 2. 总奖励分布
    plt.subplot(2, 2, 2)
    total_rewards = [res["total_reward"] for res in results]
    plt.hist(total_rewards, bins=10, alpha=0.7)
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.title("Distribution of Total Rewards")
    
    # 3. 分子复杂性分析
    plt.subplot(2, 2, 3)
    atom_counts = [mol.GetNumAtoms() for mol in molecules if mol]
    plt.hist(atom_counts, bins=15, alpha=0.7)
    plt.xlabel("Number of Atoms")
    plt.ylabel("Frequency")
    plt.title("Molecular Complexity")
    
    # 4. 分子可视化（最后一步的分子）
    plt.subplot(2, 2, 4)
    final_molecules = [res["molecules"][-1] for res in results if res["molecules"]]
    plot_molecules(final_molecules, mols_per_row=3)
    plt.title("Final Molecules from Episodes")
    
    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    plt.show()

if __name__ == "__main__":
    evaluate_agent("./models/ppo_co2rr_final.zip", num_episodes=5)
