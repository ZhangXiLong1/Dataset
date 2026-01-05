import time
import os
from contextlib import nullcontext
from copy import deepcopy

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.fusion_model import PIAFusion  # 融合 / EPIAFusion 网络
from ultralytics import YOLO               # YOLOv8
from ultralytics.utils.torch_utils import get_num_params, get_flops

try:
    import thop
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. FLOPs calculation for fusion model will be skipped.")
    print("Install with: pip install thop")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # psutil是可选的，仅用于CPU内存统计


# ======== 在这里配置你的测试图片和模型路径 ========
# 一对配准的可见光 + 红外测试图像
VIS_IMAGE_PATH = r"F:\dataset\LLVIP\LLVIP\visible\LLVIPVisTest\220120.jpg"
IR_IMAGE_PATH = r"F:\dataset\LLVIP\LLVIP\infrared\LLVIPIRTest\220120.jpg"

# YOLOv8 权重文件（例如你训练好的 best.pt）
YOLO_WEIGHTS = r"F:\YOLO\ultralytics-main\runs\train\YOLOv8-C2f-Faster LLVIP\weights\best.pt"

# 融合网络输入分辨率（越小越快）
FUSION_H, FUSION_W = 256, 256
# YOLO 检测分辨率（可以与训练时一致，比如 512 或 640）
YOLO_SIZE = 640
# ===============================================


def load_pair(device: torch.device):
    """加载一对可见光 + 红外图像，并缩放到 FUSION_H×FUSION_W，转成 [1,1,H,W] 的张量。"""
    transform = transforms.Compose([
        transforms.Resize((FUSION_H, FUSION_W)),
        transforms.ToTensor(),
    ])

    vis_img = Image.open(VIS_IMAGE_PATH).convert('L')  # 灰度 / Y 通道
    ir_img = Image.open(IR_IMAGE_PATH).convert('L')

    y_vi = transform(vis_img).unsqueeze(0).to(device)  # [1,1,H,W]
    ir = transform(ir_img).unsqueeze(0).to(device)     # [1,1,H,W]

    if device.type == 'cuda':
        y_vi = y_vi.half()
        ir = ir.half()
    return y_vi, ir


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 1. 加载融合模型
    fusion_model = PIAFusion().to(device)
    # 使用 strict=False 以兼容包含 CSAM 等新模块的结构，忽略旧权重中缺失的键
    state_dict = torch.load('pretrained/fusion_model_epoch_29.pth', map_location=device)
    fusion_model.load_state_dict(state_dict, strict=False)
    fusion_model.eval()

    # 使用半精度加速（仅在 CUDA 上启用）
    if device.type == 'cuda':
        fusion_model.half()

    # 2. 加载 YOLOv8 检测模型
    yolo_model = YOLO(YOLO_WEIGHTS)
    yolo_model.to(device)
    yolo_model.model.eval()
    if device.type == 'cuda':
        yolo_model.model.half()

    # 3. 加载一对测试图像
    y_vi, ir = load_pair(device)

    # 根据设备选择是否使用 autocast（混合精度）
    amp_ctx = torch.cuda.amp.autocast if device.type == 'cuda' else nullcontext

    # 4. 预热（不计入统计）
    warmup_iters = 20
    with torch.no_grad(), amp_ctx():
        for _ in range(warmup_iters):
            fused = fusion_model(y_vi, ir)          # [1,1,H,W]
            fused_3c = fused.repeat(1, 3, 1, 1)     # YOLOv8 需要 3 通道
            fused_3c_up = F.interpolate(
                fused_3c, size=(YOLO_SIZE, YOLO_SIZE),
                mode='bilinear', align_corners=False
            )
            _ = yolo_model.predict(fused_3c_up, imgsz=YOLO_SIZE, verbose=False)

    iters = 100

    # 5. 仅融合耗时和内存占用
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad(), amp_ctx():
        for _ in range(iters):
            _ = fusion_model(y_vi, ir)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        fusion_memory_MB = torch.cuda.max_memory_allocated() / 1e6  # 转换为MB
        torch.cuda.reset_peak_memory_stats()
    else:
        # CPU内存统计（使用psutil如果可用）
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            fusion_memory_MB = process.memory_info().rss / 1e6
        else:
            fusion_memory_MB = 0.0
    end = time.time()
    avg_time_fusion = (end - start) / iters
    fps_fusion = 1.0 / avg_time_fusion

    # 6. 仅 YOLO 耗时和内存占用（使用事先融合好的一帧）
    with torch.no_grad(), amp_ctx():
        fused_once = fusion_model(y_vi, ir)
        fused_3c_once = fused_once.repeat(1, 3, 1, 1)
        fused_3c_once_up = F.interpolate(
            fused_3c_once, size=(YOLO_SIZE, YOLO_SIZE),
            mode='bilinear', align_corners=False
        )
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad(), amp_ctx():
        for _ in range(iters):
            _ = yolo_model.predict(fused_3c_once_up, imgsz=YOLO_SIZE, verbose=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        yolo_memory_MB = torch.cuda.max_memory_allocated() / 1e6  # 转换为MB
        torch.cuda.reset_peak_memory_stats()
    else:
        # CPU内存统计
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            yolo_memory_MB = process.memory_info().rss / 1e6
        else:
            yolo_memory_MB = 0.0
    end = time.time()
    avg_time_yolo = (end - start) / iters
    fps_yolo = 1.0 / avg_time_yolo

    # 7. Fusion + YOLO 总耗时和内存占用
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad(), amp_ctx():
        for _ in range(iters):
            fused = fusion_model(y_vi, ir)
            fused_3c = fused.repeat(1, 3, 1, 1)
            fused_3c_up = F.interpolate(
                fused_3c, size=(YOLO_SIZE, YOLO_SIZE),
                mode='bilinear', align_corners=False
            )
            _ = yolo_model.predict(fused_3c_up, imgsz=YOLO_SIZE, verbose=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        total_memory_MB = torch.cuda.max_memory_allocated() / 1e6  # 转换为MB
    else:
        # CPU内存统计
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            total_memory_MB = process.memory_info().rss / 1e6
        else:
            total_memory_MB = 0.0
    end = time.time()
    avg_time_total = (end - start) / iters
    fps_total = 1.0 / avg_time_total

    # 8. 计算模型参数量和FLOPs
    print('\n' + '='*60)
    print('Model Statistics')
    print('='*60)
    
    # 8.1 融合模型统计
    fusion_params = sum(p.numel() for p in fusion_model.parameters())
    fusion_params_M = fusion_params / 1e6  # 转换为百万单位
    
    fusion_flops_G = 0.0
    if THOP_AVAILABLE:
        try:
            # 融合模型输入是两个张量 [1,1,H,W]
            fusion_model_copy = deepcopy(fusion_model)
            fusion_model_copy.eval()
            fusion_model_copy = fusion_model_copy.float()  # thop需要float32
            
            dummy_vis = torch.randn(1, 1, FUSION_H, FUSION_W).to(device)
            dummy_ir = torch.randn(1, 1, FUSION_H, FUSION_W).to(device)
            
            flops, _ = thop.profile(fusion_model_copy, inputs=(dummy_vis, dummy_ir), verbose=False)
            fusion_flops_G = flops / 1e9  # 转换为GFLOPs (10^9)
        except Exception as e:
            print(f'Warning: Failed to calculate fusion model FLOPs: {e}')
    
    print(f'\nFusion Model (PIAFusion):')
    print(f'  Params: {fusion_params_M:.2f} M ({fusion_params:,} total)')
    print(f'  FLOPs:  {fusion_flops_G:.2f} GFLOPs (input size: {FUSION_H}x{FUSION_W})')
    
    # 8.2 YOLO模型统计
    yolo_params = get_num_params(yolo_model.model)
    yolo_params_M = yolo_params / 1e6  # 转换为百万单位
    
    yolo_flops_G = get_flops(yolo_model.model, imgsz=YOLO_SIZE)  # 已经返回GFLOPs
    
    print(f'\nYOLO Model:')
    print(f'  Params: {yolo_params_M:.2f} M ({yolo_params:,} total)')
    print(f'  FLOPs:  {yolo_flops_G:.2f} GFLOPs (input size: {YOLO_SIZE}x{YOLO_SIZE})')
    
    # 8.3 总统计
    total_params_M = fusion_params_M + yolo_params_M
    total_flops_G = fusion_flops_G + yolo_flops_G
    
    print(f'\nTotal (Fusion + YOLO):')
    print(f'  Params: {total_params_M:.2f} M')
    print(f'  FLOPs:  {total_flops_G:.2f} GFLOPs')
    
    # 9. 使用融合后的图像做一次检测，并保存带框结果图
    with torch.no_grad(), amp_ctx():
        detect_results = yolo_model.predict(
            fused_3c_once_up, imgsz=YOLO_SIZE, verbose=False
        )
    save_dir = os.path.join('runs', 'benchmark_fusion')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '220120_fused_det.jpg')
    # ultralytics 的 Results 对象支持直接保存可视化结果
    detect_results[0].save(filename=save_path)

    print('\n' + '='*60)
    print('Performance Results')
    print('='*60)
    print(f'Input image paths: ')
    print(f'  VIS: {VIS_IMAGE_PATH}')
    print(f'  IR : {IR_IMAGE_PATH}')
    print(f'Fusion input size: {FUSION_H}x{FUSION_W}')
    print(f'YOLO detect size:  {YOLO_SIZE}x{YOLO_SIZE}')
    print(f'\nFusion only: {avg_time_fusion * 1000:.2f} ms  | FPS: {fps_fusion:.2f}  | Memory: {fusion_memory_MB:.2f} MB')
    print(f'YOLO only:   {avg_time_yolo * 1000:.2f} ms  | FPS: {fps_yolo:.2f}  | Memory: {yolo_memory_MB:.2f} MB')
    print(f'Fusion+YOLO: {avg_time_total * 1000:.2f} ms  | FPS: {fps_total:.2f}  | Memory: {total_memory_MB:.2f} MB')
    
    # 显示内存统计摘要
    if device.type == 'cuda':
        print(f'\nMemory Statistics (CUDA):')
        print(f'  Fusion model memory: {fusion_memory_MB:.2f} MB')
        print(f'  YOLO model memory:   {yolo_memory_MB:.2f} MB')
        print(f'  Total memory:        {total_memory_MB:.2f} MB')
        # 显示当前CUDA内存使用情况
        current_memory_MB = torch.cuda.memory_allocated() / 1e6
        reserved_memory_MB = torch.cuda.memory_reserved() / 1e6
        print(f'  Current allocated:   {current_memory_MB:.2f} MB')
        print(f'  Current reserved:    {reserved_memory_MB:.2f} MB')
    
    print(f'\nDetection image saved to: {save_path}')


if __name__ == '__main__':
    main()