from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import random

# 预定义的分子片段库 - 模拟扩散模型生成能力
FRAGMENT_LIBRARY = [
    'C', 'CC', 'CCC', 'CCO', 'O', 'CO', 'C=O', 'C=C', 'C#N', 
    'N', 'CN', 'C[N+]', 'C[O-]', 'c1ccccc1', 'C1CC1', 'C1CCC1',
    'C1NCC1', 'C1OC1', 'C(=O)O', 'C(=O)N', 'C#N', 'N=O'
]

def generate_initial_candidate(bias_toward_target=None, strength=0.5, temperature=0.5):
    """生成初始分子候选（模拟扩散模型输出）"""
    try:
        # 有一定概率直接返回目标分子（模拟强化学习引导效果）
        if bias_toward_target and random.random() < strength:
            return Chem.MolFromSmiles(bias_toward_target)
        
        # 温度控制：影响随机性
        complexity = int(1 + temperature * 4)  # 分子复杂度：1-5个片段
        
        # 组合分子片段
        fragments = random.sample(FRAGMENT_LIBRARY, complexity)
        combined_smiles = '.'.join(fragments)
        
        # 尝试组合成一个分子
        combined_mol = Chem.MolFromSmiles(combined_smiles)
        if combined_mol is None:
            # 组合失败时返回随机片段
            return Chem.MolFromSmiles(random.choice(FRAGMENT_LIBRARY))
        
        # 随机添加/修改一些原子（增加多样性）
        if random.random() < 0.3:
            return _random_modification(combined_mol)
        
        return combined_mol
    except:
        # 生成失败时返回简单分子
        return Chem.MolFromSmiles('C')

def _random_modification(mol):
    """对分子进行随机修改"""
    # 简化版：实际扩散模型会有更复杂的操作
    rand_val = random.random()
    
    if rand_val < 0.3:
        # 添加原子
        atom_types = ['C', 'O', 'N']
        new_atom = Chem.Atom(random.choice(atom_types))
        mol = Chem.AddAtom(mol, new_atom)
    elif rand_val < 0.6:
        # 添加键
        atoms = list(range(mol.GetNumAtoms()))
        if len(atoms) > 1:
            i, j = random.sample(atoms, 2)
            bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]
            mol = Chem.AddBond(mol, int(i), int(j), random.choice(bond_types))
    
    return mol
