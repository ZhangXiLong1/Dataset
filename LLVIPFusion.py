# -*- coding: utf-8 -*-
"""
LLVIP_FUSION.py
用于在 YOLOv8 训练阶段，将 LLVIP 可见光 + 红外图像
通过 PIAFusion 在线融合为 3 通道输入，再送入 YOLOv8。

数据路径严格按照你的配置：
F:/YOLO/ultralytics-main/dataset/images/
 ├─ train/LLVIPVisTrain
 ├─ train/LLVIPIRTrain
 ├─ test/LLVIPVisTest
 └─ test/LLVIPIRTest
"""
import torch
from fusion_model import PIAFusion

# -----------------------------
# 1. 全局加载 PIAFusion（只加载一次）
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fusion_model = PIAFusion().to(device)
fusion_model.load_state_dict(torch.load('./pretrained/fusion_model_epoch_29.pth', map_location=device))
fusion_model.eval()

# 冻结参数（非常重要）
for p in fusion_model.parameters():
    p.requires_grad = False


import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

# =======================
# 工具函数
# =======================

def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    return canvas


def load_labels(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r') as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            labels.append([cls, x, y, w, h])
    return np.array(labels, dtype=np.float32)


# =======================
# LLVIP Fusion Dataset
# =======================

class LLVIPFusionDataset(Dataset):
    def __init__(self, imgsz=640, mode='train', fusion_model=None):
        assert fusion_model is not None, 'PIAFusion 模型不能为空'

        self.imgsz = imgsz
        self.mode = mode
        self.fusion_model = fusion_model

        root = 'F:/YOLO/ultralytics-main/dataset/images'

        if mode == 'train':
            self.vis_dir = f'{root}/train/LLVIPVisTrain'
            self.ir_dir  = f'{root}/train/LLVIPIRTrain'
            self.label_dir = 'F:/YOLO/ultralytics-main/dataset/labels/train'
        else:
            self.vis_dir = f'{root}/test/LLVIPVisTest'
            self.ir_dir  = f'{root}/test/LLVIPIRTest'
            self.label_dir = 'F:/YOLO/ultralytics-main/dataset/labels/test'

        self.images = sorted(os.listdir(self.vis_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        vis_path = os.path.join(self.vis_dir, name)
        ir_path = os.path.join(self.ir_dir, name)
        label_path = os.path.join(self.label_dir, name.replace('.jpg', '.txt'))

        vis = cv2.imread(vis_path)
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        vis = letterbox(cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY), self.imgsz)  # 转灰度
        ir = letterbox(ir, self.imgsz)

        vis = torch.from_numpy(vis).unsqueeze(0).float() / 255.0  # [1,H,W]
        ir = torch.from_numpy(ir).unsqueeze(0).float() / 255.0  # [1,H,W]

        with torch.no_grad():
            fused = self.fusion_model(vis.unsqueeze(0).to(device), ir.unsqueeze(0).to(device))

        fused = fused.squeeze(0).cpu().repeat(3, 1, 1)  # [3,H,W]

        labels = load_labels(label_path)
        labels = torch.from_numpy(labels) if len(labels) else torch.zeros((0, 5))

        return fused, labels
