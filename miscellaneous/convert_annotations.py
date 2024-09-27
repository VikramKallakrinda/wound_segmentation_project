import os
import json
import cv2
import numpy as np

def create_mask_from_shapes(img_shape, shapes):
    """
    Creates a mask image from shapes.
    
    Args:
    - img_shape (tuple): Shape of the image (height, width).
    - shapes (list): List of shape definitions, each with a list of points.
    
    Returns:
    - np.ndarray: Mask image.
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    
    for shape in shapes:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=1)
    
    return mask

def convert_labelme_to_mask(json_dir, mask_dir):
    """
    Converts LabelMe JSON annotations to mask images.
    
    Args:
    - json_dir (str): Directory containing LabelMe JSON files.
    - mask_dir (str): Directory to save the resulting mask images.
    """
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Get image dimensions from JSON
            img_shape = (data['imageHeight'], data['imageWidth'])
            
            # Filter out invalid shapes
            valid_shapes = [shape for shape in data['shapes'] if len(shape['points']) > 2]
            
            if not valid_shapes:
                print(f"No valid shapes found in {json_file}")
                continue
            
            # Create mask from valid shapes in JSON data
            mask = create_mask_from_shapes(img_shape, valid_shapes)

            # Save mask image
            mask_path = os.path.join(mask_dir, json_file.replace('.json', '_mask.png'))
            cv2.imwrite(mask_path, mask * 255)  # Multiply by 255 to make mask visible
            print(f"Mask saved to {mask_path}")

# Directories
json_dir = r'C:\Users\vikra\wound_segmentation_env\wound_segmentation_project\data\json masks'
mask_dir = r'C:\Users\vikra\wound_segmentation_env\wound_segmentation_project\data\masks'

convert_labelme_to_mask(json_dir, mask_dir)
