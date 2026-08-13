import os
import sys

base = "/Users/thelucifer/Library/Mobile Documents/com~apple~CloudDocs/Hackathon/Plant Disease"
out_path = "val_out.txt"

with open(out_path, "w") as f:
    f.write("STEP 1: Script started\n")
    f.flush()

try:
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    with open(out_path, "a") as f:
        f.write("STEP 2: Imports succeeded\n")
        f.flush()

    m_path = os.path.join(base, 'best_model_optimized.keras')
    with open(out_path, "a") as f:
        f.write(f"STEP 3: Loading model from {m_path}\n")
        f.flush()

    model = tf.keras.models.load_model(m_path)

    with open(out_path, "a") as f:
        f.write("STEP 4: Model loaded successfully\n")
        f.flush()

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

    def validate(img_path):
        img = Image.open(img_path).convert('RGB')
        arr = np.expand_dims(np.array(img.resize((128, 128)), dtype=np.float32), axis=0)
        pred = model.predict(arr)
        h, s, v = rgb_to_hsv_numpy(np.array(img))
        mask = ((h >= 65) & (h <= 165) & (s >= 0.15) & (v >= 0.15)) | ((h >= 20) & (h < 65) & (s >= 0.15) & (v >= 0.15))
        ratio = np.sum(mask) / float(img.size[0] * img.size[1])
        top1 = float(np.max(pred[0]))
        is_valid = (ratio >= 0.12) and (top1 >= 0.65)
        return is_valid, ratio, top1

    v1, r1, t1 = validate(os.path.join(base, 'test', 'test', 'AppleScab1.JPG'))
    v2, r2, t2 = validate(os.path.join(base, 'Background_image.jpeg'))

    with open(out_path, "a") as f:
        f.write(f"TEST 1 (Leaf Image): Valid={v1}, Foliage Ratio={r1:.3f}, Top Confidence={t1:.3f}\n")
        f.write(f"TEST 2 (Non-Leaf Image): Valid={v2}, Foliage Ratio={r2:.3f}, Top Confidence={t2:.3f}\n")
        f.flush()

except Exception as err:
    with open(out_path, "a") as f:
        f.write(f"ERROR: {err}\n")
        f.flush()
