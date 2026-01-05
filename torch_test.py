import torch

# 核心验证：PyTorch版本+CUDA是否可用
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 若CUDA可用，打印显卡信息
if torch.cuda.is_available():
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ 仅CPU模式（若有NVIDIA显卡，检查驱动/CUDA安装）")