import json

path = "/Users/thelucifer/Library/Mobile Documents/com~apple~CloudDocs/Hackathon/Plant Disease/trainning_module.ipynb"

with open(path, "r", encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])

start_idx = -1
end_idx = -1

for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        if 'cnn = tf.keras.models.Sequential()' in source:
            start_idx = i
            
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'markdown':
        source = "".join(cell.get('source', []))
        if 'Compilling model' in source in source:
            end_idx = i
            break

if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
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
        "    base_model,\n",
        "    tf.keras.layers.GlobalAveragePooling2D(),\n",
        "    tf.keras.layers.Dense(256, activation='relu'),\n",
        "    tf.keras.layers.Dropout(0.3),\n",
        "    tf.keras.layers.Dense(38, activation='softmax')\n",
        "])\n"
    ]
    
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "transfer_learning_model",
        "metadata": {},
        "outputs": [],
        "source": new_source
    }
    
    # Replace cells from start_idx to end_idx - 1 with our new_cell
    del cells[start_idx:end_idx]
    cells.insert(start_idx, new_cell)
    
with open(path, "w", encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook updated for Transfer Learning!")
