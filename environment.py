import gym
from gym import spaces
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from diffusion_sampler import generate_initial_candidate

class CO2RRMaterialEnv(gym.Env):
    """电催化CO2RR材料设计强化学习环境"""
    
    def __init__(self, target_smiles='CCO', max_steps=20):
        super(CO2RRMaterialEnv, self).__init__()
        
        # 目标分子（用于计算奖励）
        self.target_smiles = target_smiles
        self.target_mol = Chem.MolFromSmiles(target_smiles)
        self.target_fp = self._get_fingerprint(self.target_mol)
        
        # 状态空间：分子指纹（2048维）
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2048,), dtype=np.float32
        )
        
        # 动作空间：连续动作（3维）
        # [动作1: 调整采样温度, 动作2: 调整多样性, 动作3: 调整约束强度]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        
        self.max_steps = max_steps
        self.current_step = 0
        self.current_mol = None
        self.current_fp = None
        
        self.reset()

    def _get_fingerprint(self, mol):
        """将分子转换为Morgan指纹"""
        if mol is None:
            return np.zeros(2048)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    
    def _calculate_reward(self, mol):
        """计算奖励函数 - 基于与目标分子的相似度"""
        if mol is None:
            return -1.0
        
        fp = self._get_fingerprint(mol)
        similarity = np.sum(fp & self.target_fp) / np.sum(fp | self.target_fp)
        
        # 额外奖励：分子复杂性（原子数）
        atom_count = mol.GetNumAtoms() if mol else 0
        complexity_bonus = min(atom_count / 20, 0.5)  # 最多奖励0.5
        
        return similarity + complexity_bonus

    def reset(self):
        """重置环境状态"""
        self.current_mol = generate_initial_candidate()
        self.current_fp = self._get_fingerprint(self.current_mol)
        self.current_step = 0
        return self.current_fp

    def step(self, action):
        """执行一个动作"""
        # 确保动作在合法范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 根据动作调整分子生成参数
        adjusted_mol = self._adjust_molecule(action)
        
        # 计算奖励
        reward = self._calculate_reward(adjusted_mol)
        
        # 更新状态
        self.current_mol = adjusted_mol
        self.current_fp = self._get_fingerprint(adjusted_mol)
        
        # 检查是否结束
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            "smiles": Chem.MolToSmiles(adjusted_mol) if adjusted_mol else "Invalid",
            "similarity": reward
        }
        
        return self.current_fp, reward, done, info

    def _adjust_molecule(self, action):
        """根据动作调整分子"""
        try:
            # 动作[0]: 调整采样温度（影响探索/利用）
            temperature = 0.5 + 0.5 * action[0]
            
            # 动作[1]: 调整多样性参数
            diversity = 0.5 + 0.5 * action[1]
            
            # 动作[2]: 调整约束强度
            constraint = 0.5 + 0.5 * action[2]
            
            # 在实际应用中，这些参数会影响扩散模型采样
            # 这里我们模拟这个效果：随机调整分子但偏向目标
            return generate_initial_candidate(
                bias_toward_target=self.target_smiles,
                strength=constraint,
                temperature=temperature
            )
        except:
            return None  # 生成失败时返回None

    def render(self, mode='human'):
        """可视化当前分子"""
        from utils import draw_molecule
        if self.current_mol:
            img = draw_molecule(self.current_mol)
            if mode == 'human':
                img.show()
            return img
        return None
