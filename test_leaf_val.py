import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

base_dir = "/Users/thelucifer/Library/Mobile Documents/com~apple~CloudDocs/Hackathon/Plant Disease"
output_file = os.path.join(base_dir, "test_results.txt")

def log(msg):
    with open(output_file, "a") as f:
        f.write(str(msg) + "\n")
        f.flush()
        os.fsync(f.fileno())

with open(output_file, "w") as f:
    f.write("=== STARTING PURE NUMPY LEAF VALIDATION TEST ===\n")
    f.flush()
    os.fsync(f.fileno())

try:
    model_path = os.path.join(base_dir, 'best_model_optimized.keras')
    log(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    log("Model loaded successfully.")

    def rgb_to_hsv_numpy(rgb_array):
        rgb = rgb_array.astype(np.float32) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        max_c = np.max(rgb, axis=2)
        min_c = np.min(rgb, axis=2)
        delta = max_c - min_c
        
        h = np.zeros_like(max_c)
        mask_delta = delta > 1e-5
        
        mask_r = mask_delta & (max_c == r)
        h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-8)) % 6.0
        
        mask_g = mask_delta & (max_c == g)
        h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-8)) + 2.0
        
        mask_b = mask_delta & (max_c == b)
        h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-8)) + 4.0
        
        h = (h * 60.0) % 360.0
        s = np.zeros_like(max_c)
        s[max_c > 1e-5] = delta[max_c > 1e-5] / max_c[max_c > 1e-5]
        v = max_c
        
        return h, s, v

    def validate_leaf_image(pil_image, prediction):
        img_rgb = np.array(pil_image.convert('RGB'))
        h, s, v = rgb_to_hsv_numpy(img_rgb)

        mask_green = (h >= 65) & (h <= 165) & (s >= 0.15) & (v >= 0.15)
        mask_brown = (h >= 20) & (h < 65) & (s >= 0.15) & (v >= 0.15)

        foliage_mask = mask_green | mask_brown
        total_pixels = float(img_rgb.shape[0] * img_rgb.shape[1])
        foliage_pixels = float(np.sum(foliage_mask))
        foliage_ratio = foliage_pixels / total_pixels

        sorted_probs = np.sort(prediction[0])[::-1]
        top1 = float(sorted_probs[0])
        top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = top1 - top2

        if foliage_ratio < 0.12:
            return False, f"Invalid Image: Non-leaf photo detected (Foliage ratio: {round(foliage_ratio * 100, 1)}% is below 12% minimum threshold)."

        if top1 < 0.65:
            return False, f"Uncertain Diagnosis: Confidence ({round(top1 * 100, 1)}%) is below 65% minimum threshold for a valid leaf."

        if margin < 0.10:
            return False, f"Ambiguous Image: Prediction split across multiple classes (Margin: {round(margin * 100, 1)}%)."

        return True, "Valid leaf image."

    def test_image(img_path):
        pil_img = Image.open(img_path)
        resized = pil_img.resize((128, 128))
        arr = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)
        pred = model.predict(arr)
        is_valid, msg = validate_leaf_image(pil_img, pred)
        return is_valid, msg, np.argmax(pred[0]), float(np.max(pred[0]))

    leaf_path = os.path.join(base_dir, 'test', 'test', 'AppleScab1.JPG')
    v1, m1, idx1, conf1 = test_image(leaf_path)
    log(f"\n[TEST 1 - Valid Leaf Image 'AppleScab1.JPG']\nValid: {v1}\nMsg: {m1}\nClass Index: {idx1}\nConfidence: {round(conf1*100, 2)}%")

    non_leaf_path = os.path.join(base_dir, 'Background_image.jpeg')
    v2, m2, idx2, conf2 = test_image(non_leaf_path)
    log(f"\n[TEST 2 - Non-Leaf Image 'Background_image.jpeg']\nValid: {v2}\nMsg: {m2}\nClass Index: {idx2}\nConfidence: {round(conf2*100, 2)}%")

    log("\n=== TEST COMPLETED SUCCESSFULLY ===")

except Exception as e:
    log(f"EXCEPTION: {e}")
    import traceback
    log(traceback.format_exc())
