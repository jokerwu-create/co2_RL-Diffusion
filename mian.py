# 电催化材料筛选系统 - 极简版
print("🎯 电催化CO2还原材料AI筛选系统启动")

def simple_material_search():
    """基础材料筛选演示"""
    materials = ["Cu纳米线", "Ag多孔膜", "Au@Pt核壳", "ZnO量子点"]
    print("\n候选材料:", materials)
    
    # 用户选择材料
    choice = input("请选择要优化的材料编号 (1-4): ")
    selected = materials[int(choice)-1]
    
    # 模拟AI优化
    print(f"\n🔬 正在用强化学习优化 {selected}...")
    print("生成扩散模型结构...")
    print("计算吸附能: -0.78eV → -0.92eV")
    print("CO选择性: 73% → 89%")
    
    return f"优化结果: {selected}_optimized"

if __name__ == "__main__":
    result = simple_material_search()
    print(f"\n✅ 优化完成！推荐材料: {result}")
