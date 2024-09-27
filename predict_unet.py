import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import os
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('unet_model.keras')

def preprocess_image(image_path, target_size=(256, 256)):
    image = load_img(image_path, color_mode='grayscale')
    image = img_to_array(image)
    image = tf.image.resize(image, target_size)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def postprocess_mask(mask):
    mask = np.squeeze(mask)  # Remove batch dimension
    mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask
    return mask

def save_comparison_images(original_image, predicted_mask, save_path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    plt.axis('off')

    plt.savefig(save_path)
    plt.close()

def predict():
    test_image_dir = 'data/images/test/original'
    output_mask_dir = 'outputs/testing/testing_results'
    output_comparison_dir = 'outputs/testing/testing_comparision'
    
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_comparison_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(test_image_dir) if f.endswith('.png')])
    
    for image_file in image_files:
        image_path = os.path.join(test_image_dir, image_file)
        original_image = load_img(image_path, color_mode='grayscale')
        original_image_array = img_to_array(original_image)
        
        processed_image = preprocess_image(image_path)
        predicted_mask = model.predict(processed_image)
        predicted_mask = postprocess_mask(predicted_mask)
        
        # Save the predicted mask
        save_img(os.path.join(output_mask_dir, image_file.replace('.png', '_mask.png')), predicted_mask)
        
        # Save comparison images
        save_comparison_images(original_image_array, predicted_mask,
                               os.path.join(output_comparison_dir, image_file.replace('.png', '_comparison.png')))

if __name__ == "__main__":
    predict()
