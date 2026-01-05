import warnings, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ===================== 1. 环境配置 =====================
os.environ["YOLO_DISABLE_AUTO_DOWNLOAD"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== 2. PIAFusion核心模型 =====================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PIAFusion(nn.Module):
    def __init__(self, in_channels=1, mid_channels=64, out_channels=1):
        super(PIAFusion, self).__init__()
        self.enc1 = ConvBlock(in_channels, mid_channels, 3, 1, 1)
        self.enc2 = ConvBlock(mid_channels, mid_channels * 2, 3, 2, 1)
        self.enc3 = ConvBlock(mid_channels * 2, mid_channels * 4, 3, 2, 1)

        self.fusion_conv1 = ConvBlock(mid_channels * 4, mid_channels * 2, 3, 1, 1)
        self.fusion_conv2 = ConvBlock(mid_channels * 2, mid_channels, 3, 1, 1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.out_conv = nn.Conv2d(mid_channels, out_channels, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def gradient(self, x):
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
            x.device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
            x.device)
        grad_x = F.conv2d(x, kernel_x, padding=1)
        grad_y = F.conv2d(x, kernel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2)

    def forward(self, vis_y, ir):
        vis_feat1 = self.enc1(vis_y)
        vis_feat2 = self.enc2(vis_feat1)
        vis_feat3 = self.enc3(vis_feat2)

        ir_feat1 = self.enc1(ir)
        ir_feat2 = self.enc2(ir_feat1)
        ir_feat3 = self.enc3(ir_feat2)

        vis_grad = self.gradient(vis_y)
        ir_grad = self.gradient(ir)
        grad_weight = ir_grad / (vis_grad + ir_grad + 1e-8)

        fusion_feat3 = grad_weight * ir_feat3 + (1 - grad_weight) * vis_feat3
        fusion_feat3 = self.fusion_conv1(fusion_feat3)

        fusion_feat2 = self.up1(fusion_feat3) + grad_weight * ir_feat2 + (1 - grad_weight) * vis_feat2
        fusion_feat2 = self.fusion_conv2(fusion_feat2)

        fusion_feat1 = self.up2(fusion_feat2) + grad_weight * ir_feat1 + (1 - grad_weight) * vis_feat1

        out = self.out_conv(fusion_feat1)
        out = self.sigmoid(out)
        return torch.clamp(out, 0, 1)


# ===================== 3. 初始化PIAFusion =====================
def init_fusion_model(fusion_weight_path=None):
    fusion_model = PIAFusion()
    if fusion_weight_path and os.path.exists(fusion_weight_path):
        fusion_model.load_state_dict(torch.load(fusion_weight_path, map_location=device))
    fusion_model = fusion_model.to(device).eval()
    return fusion_model


# ===================== 4. 自定义数据集（修复路径拼接） =====================
class FusionLLVIPDataset(Dataset):
    def __init__(self, data_root, split='train', imgsz=640):
        """
        修复路径拼接：
        - train → images/train/LLVIPVisTrain
        - val → images/test/LLVIPVisTest（你的验证集实际在test目录）
        """
        self.data_root = Path(data_root)
        self.imgsz = imgsz
        self.split = split

        # 核心修复：按你的实际路径拼接
        if split == 'train':
            # 训练集路径
            self.vis_dir = self.data_root / 'images' / 'train' / 'LLVIPVisTrain'
            self.ir_dir = self.data_root / 'images' / 'train' / 'LLVIPIRTrain'
            self.label_dir = self.data_root / 'labels' / 'train' / 'LLVIPVisTrain'
        else:
            # 验证集路径（实际在test目录）
            self.vis_dir = self.data_root / 'images' / 'test' / 'LLVIPVisTest'
            self.ir_dir = self.data_root / 'images' / 'test' / 'LLVIPIRTest'
            self.label_dir = self.data_root / 'labels' / 'test' / 'LLVIPVisTest'

        # 遍历可见光图像
        self.vis_files = list(self.vis_dir.glob('*.jpg')) + list(self.vis_dir.glob('*.png'))
        assert len(self.vis_files) > 0, f"未找到{split}集图像：{self.vis_dir}\n请检查路径是否正确！"

        # 初始化融合模型
        self.fusion_model = init_fusion_model()

        # 打印路径验证
        print(f"\n===== {split}集路径验证 =====")
        print(f"可见光路径：{self.vis_dir}")
        print(f"红外路径：{self.ir_dir}")
        print(f"标注路径：{self.label_dir}")
        print(f"图像数量：{len(self.vis_files)}")

    def __len__(self):
        return len(self.vis_files)

    def __getitem__(self, idx):
        # 1. 加载图像
        vis_path = self.vis_files[idx]
        ir_path = self.ir_dir / vis_path.name  # 红外与可见光同名
        label_path = self.label_dir / vis_path.name.replace(vis_path.suffix, '.txt')

        # 加载可见光（BGR）
        vis_img = cv2.imread(str(vis_path))
        if vis_img is None:
            raise FileNotFoundError(f"加载失败：{vis_path}")
        # 加载红外（单通道）
        ir_img = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        if ir_img is None:
            raise FileNotFoundError(f"加载失败：{ir_path}")

        # 2. PIAFusion融合
        fusion_size = 64
        # 可见光预处理
        vis_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
        vis_resized = cv2.resize(vis_gray, (fusion_size, fusion_size))
        vis_y = torch.from_numpy(vis_resized).float() / 255.0
        vis_y = vis_y.unsqueeze(0).unsqueeze(0).to(device)

        # 红外预处理
        ir_resized = cv2.resize(ir_img, (fusion_size, fusion_size))
        ir_y = torch.from_numpy(ir_resized).float() / 255.0
        ir_y = ir_y.unsqueeze(0).unsqueeze(0).to(device)

        # 融合
        with torch.no_grad():
            fused_tensor = self.fusion_model(vis_y, ir_y)

        # 3. 后处理
        fused_img = fused_tensor.squeeze(0).squeeze(0).cpu().numpy()
        fused_img = (fused_img * 255).astype(np.uint8)
        fused_img = cv2.resize(fused_img, (self.imgsz, self.imgsz))
        fused_img = cv2.cvtColor(fused_img, cv2.COLOR_GRAY2BGR)
        img = torch.from_numpy(fused_img).permute(2, 0, 1).float() / 255.0

        # 4. 加载标注
        if label_path.exists():
            labels = np.loadtxt(str(label_path), dtype=np.float32).reshape(-1, 5)
            labels = torch.from_numpy(labels)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        return img, labels


# ===================== 5. 训练函数（修复保存路径） =====================
from ultralytics import YOLO


def train_yolo():
    # 1. 初始化YOLO模型
    model = YOLO('ultralytics/cfg/models/v8/yolov8-C2f-Faster.yaml')
    model = model.to(device)
    model.train()

    # 2. 构建数据集（修复路径）
    data_root = 'F:/YOLO/ultralytics-main/dataset'
    train_dataset = FusionLLVIPDataset(data_root, split='train', imgsz=640)
    val_dataset = FusionLLVIPDataset(data_root, split='val', imgsz=640)

    # 3. 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: (torch.stack([i[0] for i in x]), [i[1] for i in x])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # 4. 优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005
    )

    # 5. 训练配置（修复保存路径）
    epochs = 100
    save_dir = Path('runs/train/YOLOv8-LLVIP-Fusion-Final')  # 自定义保存路径
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')

    # 6. 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)

            # 前向传播
            preds = model(imgs)
            # 计算损失
            loss_dict = model.loss(preds, labels)
            total_loss = sum(loss_dict.values())

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

            # 打印进度
            if batch_idx % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {total_loss.item():.4f}")

        # 平均损失
        avg_loss = train_loss / len(train_loader)
        print(f"\n=== Epoch {epoch + 1} | Average Train Loss: {avg_loss:.4f} ===")

        # 保存权重（自定义路径，不覆盖）
        torch.save(model.state_dict(), save_dir / f'epoch_{epoch + 1}.pt')
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_dir / 'best.pt')
            print(f"✅ Save best model to {save_dir}/best.pt\n")
        torch.save(model.state_dict(), save_dir / 'last.pt')


# ===================== 6. 启动训练 =====================
if __name__ == '__main__':
    train_yolo()