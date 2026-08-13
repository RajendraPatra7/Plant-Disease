import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import io
import numpy as np
from PIL import Image
# TensorFlow is imported lazily inside load_model() to avoid macOS memory pressure kill
tf = None

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

PESTICIDE_RECOMMENDATIONS = {
    'Apple Scab': {
        'spray_action': 'Apply Captan or Myclobutanil fungicide at first sign of lesion.',
        'pesticide_type': 'Protectant Fungicide',
        'dosage': '2.0 - 2.5 g / Litre of water',
        'eco_tip': 'Prune affected branches to improve airflow and reduce fungal spread.'
    },
    'Black Rot': {
        'spray_action': 'Apply Copper octanoate or Captan spray promptly.',
        'pesticide_type': 'Copper Fungicide',
        'dosage': '2.0 g / Litre of water',
        'eco_tip': 'Remove mummified fruit and infected leaf litter around tree bases.'
    },
    'Cedar Rust': {
        'spray_action': 'Spray Myclobutanil or Immunox during early leaf development.',
        'pesticide_type': 'Systemic Fungicide',
        'dosage': '1.5 - 2.0 ml / Litre of water',
        'eco_tip': 'Avoid planting host juniper trees within 500 meters of apple orchards.'
    },
    'Powdery Mildew': {
        'spray_action': 'Spray Neem oil, Potassium bicarbonate, or Sulfur-based fungicide.',
        'pesticide_type': 'Bio-Fungicide / Organic Spray',
        'dosage': '3.0 - 5.0 ml / Litre of water',
        'eco_tip': 'Apply during early morning to maximize leaf surface absorption.'
    },
    'Cercospora (Gray Leaf Spot)': {
        'spray_action': 'Apply Azoxystrobin or Pyraclostrobin strobilurin fungicide.',
        'pesticide_type': 'Strobilurin Fungicide',
        'dosage': '1.5 g / Litre of water',
        'eco_tip': 'Practice crop rotation with non-host crops like soybeans or legumes.'
    },
    'Common Rust': {
        'spray_action': 'Apply Mancozeb or Chlorothalonil protectant fungicide.',
        'pesticide_type': 'Broad-Spectrum Fungicide',
        'dosage': '2.0 g / Litre of water',
        'eco_tip': 'Use spot-spraying directly on pustules to limit chemical usage.'
    },
    'Northern Leaf Blight': {
        'spray_action': 'Spray Propiconazole or Azoxystrobin upon lesion detection.',
        'pesticide_type': 'Triazole Fungicide',
        'dosage': '1.0 - 1.5 ml / Litre of water',
        'eco_tip': 'Incorporate resistant crop hybrids in subsequent planting cycles.'
    },
    'Esca (Black Measles)': {
        'spray_action': 'Apply Sodium arsenite or Fosetyl-Al soil drench.',
        'pesticide_type': 'Systemic Fungicide',
        'dosage': '2.5 ml / Litre of water',
        'eco_tip': 'Seal pruning wounds with protective sealant immediately after cutting.'
    },
    'Leaf Blight (Isariopsis)': {
        'spray_action': 'Spray Copper oxychloride or Mancozeb fungicide.',
        'pesticide_type': 'Contact Fungicide',
        'dosage': '3.0 g / Litre of water',
        'eco_tip': 'Maintain optimal canopy spacing for enhanced sun penetration.'
    },
    'Huanglongbing (Citrus Greening)': {
        'spray_action': 'Control Asian Citrus Psyllid vector using Imidacloprid or Neem spray.',
        'pesticide_type': 'Systemic Insecticide',
        'dosage': '0.5 - 1.0 ml / Litre of water',
        'eco_tip': 'Remove infected trees promptly to protect surrounding healthy citrus trees.'
    },
    'Bacterial Spot': {
        'spray_action': 'Apply Copper hydroxide mixed with Mancozeb.',
        'pesticide_type': 'Bactericide / Copper Spray',
        'dosage': '2.0 g / Litre of water',
        'eco_tip': 'Avoid overhead drip irrigation that splashes bacterial spores onto foliage.'
    },
    'Early Blight': {
        'spray_action': 'Apply Chlorothalonil or Copper-based spray every 7-10 days.',
        'pesticide_type': 'Foliar Fungicide',
        'dosage': '2.0 - 2.5 ml / Litre of water',
        'eco_tip': 'Mulch around crop base to prevent soil-borne spore splashing.'
    },
    'Late Blight': {
        'spray_action': 'Apply Metalaxyl or Dimethomorph systemic fungicide immediately.',
        'pesticide_type': 'Systemic Oomycete Fungicide',
        'dosage': '2.0 g / Litre of water',
        'eco_tip': 'Late Blight spreads rapidly in humid weather; inspect surrounding plants immediately.'
    },
    'Leaf Scorch': {
        'spray_action': 'Spray Captan or Thiram immediately post-harvest or early spring.',
        'pesticide_type': 'Fungicide Spray',
        'dosage': '2.0 g / Litre of water',
        'eco_tip': 'Remove old infected leaves post-harvest to reduce overwintering spores.'
    },
    'Leaf Mold': {
        'spray_action': 'Apply Difenoconazole or Copper hydroxide spray.',
        'pesticide_type': 'Foliar Bio-Fungicide',
        'dosage': '1.5 ml / Litre of water',
        'eco_tip': 'Reduce greenhouse relative humidity below 85% to inhibit spore germination.'
    },
    'Septoria Leaf Spot': {
        'spray_action': 'Apply Chlorothalonil or Copper fungicide spray.',
        'pesticide_type': 'Contact Fungicide',
        'dosage': '2.0 g / Litre of water',
        'eco_tip': 'Remove infected lower leaves to restrict upwards disease progression.'
    },
    'Spider Mites': {
        'spray_action': 'Apply Abamectin miticide or Insecticidal soap spray.',
        'pesticide_type': 'Acaricide / Miticide',
        'dosage': '1.0 ml / Litre of water',
        'eco_tip': 'Increase environmental humidity or spray undersides of leaves with water.'
    },
    'Target Spot': {
        'spray_action': 'Apply Azoxystrobin or Chlorothalonil protectant fungicide.',
        'pesticide_type': 'Broad-Spectrum Fungicide',
        'dosage': '2.0 g / Litre of water',
        'eco_tip': 'Ensure adequate nitrogen fertilizer balance to maintain leaf vigor.'
    },
    'Yellow Leaf Curl Virus': {
        'spray_action': 'Control Whitefly vectors using Thiamethoxam or Neem oil.',
        'pesticide_type': 'Insecticide / Vector Control',
        'dosage': '0.5 g / Litre of water',
        'eco_tip': 'Use yellow sticky traps and fine insect mesh netting around crop rows.'
    },
    'Mosaic Virus': {
        'spray_action': 'No chemical cure; spray mineral oil or insecticide to control Aphid vectors.',
        'pesticide_type': 'Vector Control Insecticide',
        'dosage': '1.0 ml / Litre of water',
        'eco_tip': 'Disinfect garden tools with 10% bleach solution between handling plants.'
    }
}

def rgb_to_hsv_numpy(rgb_array):
    """Vectorized RGB to HSV transformation in pure NumPy (0-360 Hue, 0-1 Sat, 0-1 Val)."""
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

class ModelService:
    def __init__(self):
        self.model = None
        self.model_path = self._find_model_path()
        # Do not load model at import time (Lazy loading)

    def _find_model_path(self):
        # 1. Check MODEL_PATH env variable (can be set in Render dashboard)
        env_path = os.environ.get("MODEL_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # 2. Resolve relative to project structure
        # backend/app/services/ -> ../../.. = project root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        possible_paths = [
            os.path.join(base_dir, 'best_model_optimized.keras'),   # repo root (local + Docker)
            os.path.join(base_dir, 'trained_model.keras'),
            '/app/best_model_optimized.keras',                        # Render Docker root
            os.path.join(os.getcwd(), 'best_model_optimized.keras'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def load_model(self):
        global tf
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                print(f"Loading TensorFlow model lazily from: {self.model_path}", flush=True)
                if tf is None:
                    import tensorflow as tf_module
                    tf = tf_module
                self.model = tf.keras.models.load_model(self.model_path)
                print("TensorFlow model loaded successfully.", flush=True)
            else:
                print("WARNING: Model file best_model_optimized.keras not found.", flush=True)

    def validate_leaf_image(self, pil_image: Image.Image, predictions: np.ndarray):
        """
        Validates if an uploaded image is a legitimate plant leaf using pure NumPy HSV foliage color analysis
        and Softmax confidence & margin thresholding.
        """
        img_rgb = np.array(pil_image.convert('RGB'))
        h, s, v = rgb_to_hsv_numpy(img_rgb)

        # Foliage masks:
        # Green chlorophyll spectrum: Hue 65° to 165°
        mask_green = (h >= 65) & (h <= 165) & (s >= 0.15) & (v >= 0.15)
        # Diseased / Yellow / Rust / Brown leaf spectrum: Hue 20° to 65°
        mask_brown = (h >= 20) & (h < 65) & (s >= 0.15) & (v >= 0.15)

        foliage_mask = mask_green | mask_brown
        total_pixels = float(img_rgb.shape[0] * img_rgb.shape[1])
        foliage_pixels = float(np.sum(foliage_mask))
        foliage_ratio = foliage_pixels / total_pixels

        # Probability Confidence & Top-2 Margin Checks
        sorted_probs = np.sort(predictions[0])[::-1]
        top1 = float(sorted_probs[0])
        top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = top1 - top2

        # Rejection Criteria
        if foliage_ratio < 0.12:
            return False, f"The uploaded image does not appear to be a plant leaf. (Foliage color coverage: {round(foliage_ratio * 100, 1)}% is below 12% minimum threshold)."

        if top1 < 0.65:
            return False, f"Uncertain diagnosis. Confidence score ({round(top1 * 100, 1)}%) is below 65% minimum threshold for a valid crop leaf."

        if margin < 0.10:
            return False, f"Ambiguous image. Prediction is split across multiple classes (Confidence margin: {round(margin * 100, 1)}%). Please upload a clearer leaf photo."

        return True, "Valid plant leaf image."

    def get_pesticide_info(self, raw_class_name: str):
        if raw_class_name.startswith("Healthy"):
            return {
                'spray_action': 'No chemical pesticide required.',
                'pesticide_type': 'None (Organic Care)',
                'dosage': '0 ml / Litre',
                'eco_tip': 'Crop leaf is healthy! Continue routine watering, weeding, and balanced fertilizing.'
            }
        
        # Match disease key in PESTICIDE_RECOMMENDATIONS
        for disease_key, info in PESTICIDE_RECOMMENDATIONS.items():
            if disease_key.lower() in raw_class_name.lower():
                return info
        
        return {
            'spray_action': 'Apply organic broad-spectrum bio-fungicide or neem oil spray.',
            'pesticide_type': 'General Bio-Protectant',
            'dosage': '2.0 ml / Litre of water',
            'eco_tip': 'Spot-spray only the affected plant clusters to minimize environmental impact.'
        }

    def predict(self, image_bytes: bytes):
        self.load_model()
        if self.model is None:
            raise ValueError("TensorFlow model could not be loaded.")

        # Preprocess image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        resized_image = pil_image.resize((128, 128))
        input_arr = np.array(resized_image, dtype=np.float32)
        input_arr = np.expand_dims(input_arr, axis=0)  # Shape: (1, 128, 128, 3)

        # Run inference
        predictions = self.model.predict(input_arr)

        # Execute Leaf Image Validation
        is_valid, validation_msg = self.validate_leaf_image(pil_image, predictions)

        if not is_valid:
            return {
                "is_valid_leaf": False,
                "error_message": validation_msg,
                "confidence": round(float(np.max(predictions[0])), 4),
                "confidence_percentage": round(float(np.max(predictions[0])) * 100, 2)
            }

        result_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][result_idx])

        raw_class = CLASS_NAMES[result_idx] if result_idx < len(CLASS_NAMES) else "Unknown"

        # Parse crop and disease status
        is_healthy = "Healthy" in raw_class
        parts = raw_class.split(" - ")
        if is_healthy:
            crop_name = raw_class.replace("Healthy ", "").replace(" Leaf", "").strip()
            disease_name = "Healthy (No Disease)"
            status = "Healthy"
        else:
            crop_name = parts[0].replace(" Leaf", "").strip() if len(parts) > 0 else "Crop"
            disease_name = parts[1] if len(parts) > 1 else raw_class
            status = "Diseased"

        recommendations = self.get_pesticide_info(raw_class)

        return {
            "is_valid_leaf": True,
            "class_index": result_idx,
            "full_class_name": raw_class,
            "crop": crop_name,
            "disease": disease_name,
            "status": status,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "recommendations": recommendations
        }


# Singleton instance
model_service = ModelService()
