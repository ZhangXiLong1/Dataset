import warnings
warnings.filterwarnings('ignore')
import torch, glob, tqdm
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info


if __name__ == '__main__':
    flops_dict = {}
    model_size, model_type = 'n', 'yolov8'

    if model_type == 'yolov8':
        yaml_base_path = 'ultralytics/cfg/models/v8'
        yaml_info = f'{yaml_base_path}/*.yaml'
    elif model_type == 'yolov10':
        yaml_base_path = 'ultralytics/cfg/models/v10'
        yaml_info = f'{yaml_base_path}/{model_type}{model_size}*.yaml'

    for yaml_path in tqdm.tqdm(glob.glob(yaml_info)):
        if model_type == 'yolov8':
            yaml_path = yaml_path.replace(f'{yaml_base_path}/{model_type}', f'{yaml_base_path}/{model_type}{model_size}')

        if 'DCN' in yaml_path:
            continue

        try:
            model = YOLO(yaml_path)
            model.fuse()
            n_l, n_p, n_g, flops = model_info(model.model)
            flops_dict[yaml_path] = [flops, n_p]
        except:
            continue
    
    sorted_items = sorted(flops_dict.items(), key=lambda x: x[1][0])
    for key, value in sorted_items:
        print(f"{key}: {value[0]:.2f} GFLOPs {value[1]:,} Params")