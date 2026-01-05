import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    #LLVIP Fus
    model.train(#data='F:/YOLO/ultralytics-main/dataset/LLVIP.yaml',
                data='F:/YOLO/ultralytics-main/dataset/XXX.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=32,
                close_mosaic=0,
                workers=8, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                optimizer='SGD', # using SGD
                # device='0,1', # 指定显卡和多卡训练参考<YOLOV8V10配置文件.md>下方常见错误和解决方案
                # patience=0, # set 0 to close earlystop.
                resume=True, # 断点续训,YOLO初始化时选择last.pt,例如YOLO('last.pt')
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='YOLOv8s WiderPerson',
                )