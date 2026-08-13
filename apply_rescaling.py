import json

path = "/Users/thelucifer/Library/Mobile Documents/com~apple~CloudDocs/Hackathon/Plant Disease/trainning_module.ipynb"

with open(path, "r", encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])

for cell in cells:
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        if 'base_model =' in source and 'MobileNetV2' in source:
            # Reconstruct the source to include BatchNormalization
            new_source = [
                "data_augmentation = tf.keras.Sequential([\n",
                "  tf.keras.layers.RandomFlip(\"horizontal_and_vertical\"),\n",
                "  tf.keras.layers.RandomRotation(0.2),\n",
                "  tf.keras.layers.RandomZoom(0.2),\n",
                "])\n",
                "\n",
                "# Optimization 3: Transfer Learning using MobileNetV2\n",
                "base_model = tf.keras.applications.MobileNetV2(\n",
                "    input_shape=(128, 128, 3),\n",
                "    include_top=False,\n",
                "    weights='imagenet'\n",
                ")\n",
                "base_model.trainable = False # Freeze the base model to prevent destroying the pre-trained weights\n",
                "\n",
                "cnn = tf.keras.models.Sequential([\n",
                "    tf.keras.layers.Input(shape=(128, 128, 3)),\n",
                "    data_augmentation,\n",
                "    tf.keras.layers.Rescaling(1./127.5, offset=-1), # CRITICAL FIX: Scaling pixel values to [-1, 1] for MobileNetV2\n",
                "    base_model,\n",
                "    tf.keras.layers.GlobalAveragePooling2D(),\n",
                "    tf.keras.layers.BatchNormalization(), # Added Batch Normalization here!\n",
                "    tf.keras.layers.Dense(256, activation='relu'),\n",
                "    tf.keras.layers.BatchNormalization(), # And added it here!\n",
                "    tf.keras.layers.Dropout(0.3),\n",
                "    tf.keras.layers.Dense(38, activation='softmax')\n",
                "])\n"
            ]
            cell['source'] = new_source
            break
            
with open(path, "w", encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Rescaling added successfully!")
