import json

path = "/Users/thelucifer/Library/Mobile Documents/com~apple~CloudDocs/Hackathon/Plant Disease/trainning_module.ipynb"

# Using standard JSON library since ipynb files are simply JSON structures
with open(path, "r", encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        # Join lines for easier evaluation
        lines = cell.get('source', [])
        joined_source = "".join(lines)
        
        # 1. Pipeline optimization (only if not already added)
        if 'validation_set = tf.keras.utils.image_dataset_from_directory' in joined_source and 'AUTOTUNE' not in joined_source:
            joined_source += "\n\n# Optimization: tf.data Pipeline for faster I/O\nAUTOTUNE = tf.data.AUTOTUNE\ntraining_set = training_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)\nvalidation_set = validation_set.cache().prefetch(buffer_size=AUTOTUNE)"
        
        # 2. Add Data augmentation & Input Shape mapping
        if 'cnn = tf.keras.models.Sequential()' in joined_source and 'data_augmentation' not in joined_source:
            joined_source = """cnn = tf.keras.models.Sequential()

# Optimization 1: Data Augmentation Layer to prevent overfitting
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomZoom(0.2),
])

cnn.add(tf.keras.layers.Input(shape=(128, 128, 3)))
cnn.add(data_augmentation)"""
            
        # Strip input_shape from the first conv layer as we now declare Input explicitly
        if 'input_shape=[128,128,3]' in joined_source:
            joined_source = joined_source.replace(",input_shape=[128,128,3]", "").replace("input_shape=[128,128,3],", "").replace("input_shape=[128,128,3]", "")
            
        # 3. Model Optimization: Replace Flatten with GlobalAveragePooling2D
        if 'tf.keras.layers.Flatten()' in joined_source:
            joined_source = "cnn.add(tf.keras.layers.GlobalAveragePooling2D()) # Optimization: Greatly reduces parameters to avoid overfitting"
            
        # 3. Model Optimization: Reduce Dense layer params
        if 'units=1500' in joined_source:
            joined_source = "cnn.add(tf.keras.layers.Dense(units=256,activation='relu')) # Optimization: Reduced units for lighter architecture"
            
        # 4. Training Optimization: Adding Callbacks
        if 'training_history = cnn.fit' in joined_source and 'callbacks' not in joined_source:
            joined_source = """from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Optimization 2: Added Callbacks
# Early stopping halts training when validation accuracy stops improving
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
# Saves the absolute best performing model across all epochs
checkpoint = ModelCheckpoint('best_model_optimized.keras', monitor='val_accuracy', save_best_only=True)
# Shrinks learning rate dynamically if the model plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

training_history = cnn.fit(
    x=training_set, 
    validation_data=validation_set, 
    epochs=20, # We can safely use more epochs now because early stopping protects us
    callbacks=[early_stopping, checkpoint, reduce_lr]
)"""
        
        # Split source back into Keras cells array format
        new_lines = []
        split_source = joined_source.split('\n')
        for i, line in enumerate(split_source):
            if i < len(split_source) - 1:
                new_lines.append(line + '\n')
            else:
                if line: # only append last line if it's not empty, or preserve if necessary
                    new_lines.append(line)
        cell['source'] = new_lines

with open(path, "w", encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook JSON manipulated successfully!")
