import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Parameters
IMG_SIZE = (64, 64)  # Image size (height, width)
BATCH_SIZE = 32      # Batch size
EPOCHS = 15          # Number of epochs
LEARNING_RATE = 0.001

# Paths to your dataset (replace with your directories)
TRAIN_DIR = '/Users/ghost/Desktop/detect_Signs/TRAIN_DIR'
VAL_DIR = '/Users/ghost/Desktop/detect_Signs/VAL_DIR'

# Function to load and preprocess images
def preprocess_image(image_path, label, num_classes):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    image = image.resize(IMG_SIZE)  # Resize image
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    label = tf.one_hot(label, depth=num_classes)  # Convert label to one-hot
    return image, label

# Function to load dataset
def load_dataset(data_dir):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for label_name in class_names:
        label_dir = os.path.join(data_dir, label_name)

        # Skip files like `.DS_Store` and ensure only directories are processed
        if not os.path.isdir(label_dir):
            continue

        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)

            # Skip invalid files and ensure only image files are processed
            if not os.path.isfile(img_path):
                continue

            # Validate file extensions (e.g., .jpg, .jpeg, .png)
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_paths.append(img_path)
            labels.append(class_to_idx[label_name])

    num_classes = len(class_names)
    data = [preprocess_image(img, lbl, num_classes) for img, lbl in zip(image_paths, labels)]
    images, labels = zip(*data)
    return np.array(images), np.array(labels), num_classes

# Load training and validation data
train_images, train_labels, num_classes = load_dataset(TRAIN_DIR)
val_images, val_labels, _ = load_dataset(VAL_DIR)

# Create TensorFlow datasets for batching
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images)).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(BATCH_SIZE)

# Build the CNN model using TensorFlow
def build_model(num_classes):
    model = tf.keras.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        
        # Flatten the output for the fully connected layers
        tf.keras.layers.Flatten(),
        
        # Fully connected layers
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Initialize the model
model = build_model(num_classes)

# Define loss, optimizer, and accuracy metric
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

# # Training loop
# for epoch in range(EPOCHS):
#     print(f"Epoch {epoch + 1}/{EPOCHS}")
    
#     # Training step
#     for step, (x_batch, y_batch) in enumerate(train_ds):
#         with tf.GradientTape() as tape:
#             logits = model(x_batch, training=True)
#             loss = loss_fn(y_batch, logits)
        
#         # Backpropagation
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
#     # Validation step
#     val_accuracy = tf.keras.metrics.CategoricalAccuracy()
#     for x_batch, y_batch in val_ds:
#         val_logits = model(x_batch, training=False)
#         val_accuracy.update_state(y_batch, val_logits)
    
#     print(f"Validation Accuracy: {val_accuracy.result().numpy():.4f}")
    
    
# Save the trained model after training completes
model_path = '/Users/ghost/Desktop/detect_Signs/Model_path'
tf.saved_model.save(model, model_path)
print(f"Model saved to: {model_path}")
