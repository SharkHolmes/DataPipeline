import io
import os
import json
import logging
import torch
from PIL import Image
from ultralytics import YOLO

INPUT_CONTENT_TYPE = 'image/jpeg'
OUTPUT_CONTENT_TYPE = 'application/json'
logger = logging.getLogger(__name__)

# 클래스 이름 정의
CLASS_NAMES = [
    'Hammer', 'SSD', 'Alcohol', 'Spanner', 'Axe', 'Awl', 'Throwing Knife', 'Firecracker',
    'Thinner', 'Plier', 'Match', 'Smart Phone', 'Scissors', 'Tablet PC', 'Solid Fuel',
    'Bat', 'Portable Gas', 'Nail Clippers', 'Knife', 'Metal Pipe', 'Electronic Cigarettes(Liquid)',
    'Supplymentary Battery', 'Bullet', 'Gun Parts', 'USB', 'Liquid', 'Aerosol', 'Screwdriver',
    'Chisel', 'Handcuffs', 'Lighter', 'HDD', 'Electronic Cigarettes', 'Battery', 'Gun',
    'Laptop', 'Saw', 'Zippo Oil', 'Stun Gun', 'Camera', 'Camcorder', 'SD Card'
]

def model_fn(model_dir: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'loading model on device: {device}')
    
    model = YOLO(os.path.join(model_dir, 'model.pt'))
    model.to(device)
    model.model.eval()
    return model

def input_fn(request_body, request_content_type=INPUT_CONTENT_TYPE):
    logger.info('Serializing input.')
    if request_content_type == INPUT_CONTENT_TYPE:
        return Image.open(io.BytesIO(request_body))
    raise Exception('Requested unsupported ContentType in request_content_type: ' + request_content_type)

def predict_fn(input_data, model):
    logger.info('Making prediction.')
    
    with torch.no_grad():
        result = model(input_data)
    return result

def output_fn(prediction_output, accept=OUTPUT_CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == OUTPUT_CONTENT_TYPE:
        output = []
        for result in prediction_output:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                output.append({
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf),
                    'class': CLASS_NAMES[int(box.cls)]
                })
        return json.dumps(output)
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)