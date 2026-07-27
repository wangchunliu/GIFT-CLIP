# check_wandb_status_fixed.py
import wandb
import os

print("=== WandB 状态检查 ===")

# 检查环境变量
print(f"WANDB_DISABLED 环境变量: '{os.environ.get('WANDB_DISABLED', '未设置')}'")
print(f"WANDB_API_KEY 是否存在: {'是' if os.environ.get('WANDB_API_KEY') else '否'}")

try:
    # 尝试不指定mode，让WandB使用默认设置
    print("\n正在尝试初始化 WandB...")
    run = wandb.init(project="wandb-test-project", job_type="test")
    
    if run is not None:
        print("✅ WandB 初始化成功！")
        print(f"运行ID: {run.id}")
        print(f"项目: {run.project}")
        print(f"URL: {run.url}")
        run.finish()
    else:
        print("❌ WandB 初始化返回 None")
        
except Exception as e:
    print(f"❌ WandB 初始化失败: {e}")
    
print("\n=== 检查完成 ===")

