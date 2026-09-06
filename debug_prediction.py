#!/usr/bin/env python3
"""
Debug script to test predictions and show what the model actually sees
Usage: python debug_prediction.py <image_path>
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = [
    'Apple Leaf - Apple Scab',
    'Apple Leaf - Black Rot',
    'Apple Leaf - Cedar Rust',
    'Healthy Apple Leaf',
    'Healthy Blueberry Leaf',
    'Cherry Leaf - Powdery Mildew',
    'Healthy Cherry Leaf',
    'Corn Leaf - Cercospora (Gray Leaf Spot)',
    'Corn Leaf - Common Rust',
    'Corn Leaf - Northern Leaf Blight',
    'Healthy Corn Leaf',
    'Grape Leaf - Black Rot',
    'Grape Leaf - Esca (Black Measles)',
    'Grape Leaf - Leaf Blight (Isariopsis)',
    'Healthy Grape Leaf',
    'Orange Leaf - Huanglongbing (Citrus Greening)',
    'Peach Leaf - Bacterial Spot',
    'Healthy Peach Leaf',
    'Bell Pepper Leaf - Bacterial Spot',
    'Healthy Bell Pepper Leaf',
    'Potato Leaf - Early Blight',
    'Potato Leaf - Late Blight',
    'Healthy Potato Leaf',
    'Healthy Raspberry Leaf',
    'Healthy Soybean Leaf',
    'Squash Leaf - Powdery Mildew',
    'Strawberry Leaf - Leaf Scorch',
    'Healthy Strawberry Leaf',
    'Tomato Leaf - Bacterial Spot',
    'Tomato Leaf - Early Blight',
    'Tomato Leaf - Late Blight',
    'Tomato Leaf - Leaf Mold',
    'Tomato Leaf - Septoria Leaf Spot',
    'Tomato Leaf - Spider Mites',
    'Tomato Leaf - Target Spot',
    'Tomato Leaf - Yellow Leaf Curl Virus',
    'Tomato Leaf - Mosaic Virus',
    'Healthy Tomato Leaf'
]

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_prediction.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)

    print("=" * 80)
    print(f"DEBUG PREDICTION: {image_path}")
    print("=" * 80)

    # Load model
    print("\n[1] Loading model...")
    model = tf.keras.models.load_model('best_model_optimized.keras')
    print(f"✓ Model loaded: {model.input_shape} → {model.output_shape}")

    # Load and preprocess image
    print(f"\n[2] Loading image: {image_path}")
    pil_image = Image.open(image_path).convert('RGB')
    original_size = pil_image.size
    print(f"✓ Original size: {original_size}")

    resized_image = pil_image.resize((128, 128))
    print(f"✓ Resized to: 128x128")

    # Convert to array
    input_arr = np.array(resized_image, dtype=np.float32)
    print(f"✓ Array shape: {input_arr.shape}, dtype: {input_arr.dtype}")
    print(f"✓ Pixel value range: [{input_arr.min():.1f}, {input_arr.max():.1f}]")

    # The model has a Rescaling layer built-in, so we pass [0, 255] values
    input_arr = np.expand_dims(input_arr, axis=0)

    # Run prediction
    print("\n[3] Running inference...")
    predictions = model.predict(input_arr, verbose=0)

    # Get top 5 predictions
    top5_indices = np.argsort(predictions[0])[::-1][:5]

    print("\n" + "=" * 80)
    print("TOP 5 PREDICTIONS:")
    print("=" * 80)

    for i, idx in enumerate(top5_indices, 1):
        class_name = CLASS_NAMES[idx]
        confidence = predictions[0][idx] * 100
        bar_length = int(confidence / 2)
        bar = "█" * bar_length
        print(f"{i}. [{confidence:5.2f}%] {bar:<50} {class_name}")

    # Calculate margin
    sorted_probs = np.sort(predictions[0])[::-1]
    margin = (sorted_probs[0] - sorted_probs[1]) * 100

    print("\n" + "=" * 80)
    print("CONFIDENCE ANALYSIS:")
    print("=" * 80)
    print(f"Top-1 Confidence: {sorted_probs[0] * 100:.2f}%")
    print(f"Top-2 Confidence: {sorted_probs[1] * 100:.2f}%")
    print(f"Margin (Top1 - Top2): {margin:.2f}%")

    if sorted_probs[0] < 0.50:
        print("\n⚠️  WARNING: Low confidence (<50%). Model is uncertain!")
        print("   Possible reasons:")
        print("   - Image is very different from training data")
        print("   - Poor lighting or image quality")
        print("   - Unusual leaf angle or background")

    if margin < 8.0:
        print(f"\n⚠️  WARNING: Low margin ({margin:.1f}%). Prediction is ambiguous!")
        print("   The model is split between multiple classes")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
