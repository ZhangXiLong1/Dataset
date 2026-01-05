import warnings, os
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from LLVIPFusion import LLVIPFusionDataset
from fusion_model import PIAFusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# 1. 初始化 PIAFusion
# -----------------------------
fusion_model = PIAFusion().to(device)
fusion_model.load_state_dict(torch.load('./pretrained/fusion_model_epoch_29.pth', map_location=device))
fusion_model.eval()
for p in fusion_model.parameters():
    p.requires_grad = False

# -----------------------------
# 2. 初始化 Dataset & DataLoader
# -----------------------------
train_dataset = LLVIPFusionDataset(imgsz=640, mode='train', fusion_model=fusion_model)
val_dataset   = LLVIPFusionDataset(imgsz=640, mode='val', fusion_model=fusion_model)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

# -----------------------------
# 3. 初始化 YOLOv8
# -----------------------------
model = YOLO('ultralytics/cfg/models/v8/yolov8-C2f-Faster.yaml')
# model.load('yolov8n.pt')  # 如果想加载预训练权重

# -----------------------------
# 4. 训练
# -----------------------------
model.train(
    dataloader=train_loader,  # 用自定义 dataloader
    val_dataloader=val_loader,
    epochs=100,
    imgsz=640,
    optimizer='SGD',
    project='runs/train',
    name='YOLOv8-C2f-Faster LLVIP Fusion win!',
    cache=False,
    workers=8,
)
