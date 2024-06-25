import torch
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import pandas as pd
import numpy as np

# Ensure YOLOv5 is in the path
sys.path.append('/Users/pranathiprabhala/Desktop/model_final_api/yolov5')

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

def load_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights_path, device=device)
    model.eval()
    return model

def process_image(image_path, model, imgsz=416):
    dataset = LoadImages(image_path, img_size=imgsz)
    all_detections = []
    for path, img, im0s, vid_cap, *extra in dataset:
        img = torch.from_numpy(img).to(model.device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False, max_det=1000)
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
                detections = pd.DataFrame(det.cpu().numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
                detections['image_width'] = im0s.shape[1]
                detections['image_height'] = im0s.shape[0]
                all_detections.append(detections)
    return pd.concat(all_detections) if all_detections else pd.DataFrame()

def assign_teeth_indices(detections, image_width, image_height):
    # Average widths of teeth in mm
    tooth_widths = [8.5, 6.5, 7.5, 7.0, 6.0, 6.0, 10.0, 10.5, 10.5, 10.0, 6.0, 6.0, 7.0, 7.5, 6.5, 8.5]
    total_width = sum(tooth_widths)
    scale_factors = [width / total_width for width in tooth_widths]

    upper_tooth_regions = {i+1: sum(scale_factors[:i+1]) for i in range(len(scale_factors))}
    lower_tooth_regions = {i+17: sum(scale_factors[:i+1]) for i in range(len(scale_factors))}

    detections['centroid_x'] = (detections['xmin'] + detections['xmax']) / 2 / image_width
    detections['centroid_y'] = (detections['ymin'] + detections['ymax']) / 2 / image_height

    def get_tooth_index(row):
        if row['centroid_y'] < 0.5:  # Upper teeth
            return min(upper_tooth_regions, key=lambda k: abs(upper_tooth_regions[k] - row['centroid_x']))
        else:  # Lower teeth
            return min(lower_tooth_regions, key=lambda k: abs(lower_tooth_regions[k] - row['centroid_x']))

    detections['tooth_index'] = detections.apply(get_tooth_index, axis=1)
    return detections

names = ['AmalgamRestoration', 'CompositeResinRestoration', 'Crown', 'DentalCaries', 'ImpactedTooth', 
         'Overhang', 'RetainedRoots', 'RootCanalTreatment', 'PeriapicalLesion', 'Post']

# Load the model once at startup
model_path = 'best.pt'  # Adjust this to your model's path
model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_path = os.path.join('/tmp', image_file.filename)
    image_file.save(image_path)

    detections = process_image(image_path, model)
    if detections.empty:
        return jsonify({'error': 'No detections made'}), 400

    indexed_detections = assign_teeth_indices(detections, detections['image_width'].iloc[0], detections['image_height'].iloc[0])
    

    indexed_detections['class_name'] = indexed_detections['class'].apply(lambda x: names[int(x)])
    
    result = indexed_detections[['xmin', 'xmax', 'ymin', 'ymax', 'tooth_index', 'class_name', 'confidence']].to_dict(orient='records')
    
    
    total_detections = len(indexed_detections)
    print(f"Total number of detections: {total_detections}")

    return jsonify({'total_detections': total_detections, 'detections': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
