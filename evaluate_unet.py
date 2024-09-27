import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_image(image_path):
    image = Image.open(image_path)
    return img_to_array(image)

def concatenate_images(original_image, manual_mask, predicted_mask):
    original_image = np.array(original_image)
    manual_mask = np.array(manual_mask)
    predicted_mask = np.array(predicted_mask)
    
    combined_image = np.concatenate([original_image, manual_mask, predicted_mask], axis=1)
    return Image.fromarray(combined_image)

def evaluate():
    image_dir = 'data/images/train/original'
    manual_mask_dir = 'data/images/train/masks'
    predicted_mask_dir = 'outputs/training/training_predicted_masks'
    output_dir = 'outputs/training/training_comparision'
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    os.makedirs(output_dir, exist_ok=True)
    
    for image_file in image_files:
        original_image_path = os.path.join(image_dir, image_file)
        manual_mask_path = os.path.join(manual_mask_dir, image_file.replace('.png', '_mask.png'))
        predicted_mask_path = os.path.join(predicted_mask_dir, image_file.replace('.png', '_mask.png'))
        
        original_image = load_image(original_image_path)
        manual_mask = load_image(manual_mask_path)
        predicted_mask = load_image(predicted_mask_path)
        
        combined_image = concatenate_images(original_image, manual_mask, predicted_mask)
        combined_image.save(os.path.join(output_dir, image_file))

if __name__ == "__main__":
    evaluate()
