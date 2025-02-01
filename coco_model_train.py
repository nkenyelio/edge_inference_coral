import tensorflow as tf
import tensorflow_datasets as tfds

# Load COCO 2017 validation dataset
def load_coco_dataset():
    coco_data = tfds.load("coco/val2017", split="validation", as_supervised=True)
    return coco_data

# Preprocess the dataset
def preprocess_data(image, label, img_size=(224, 224)):
    # Resize images to match the model input size
    image = tf.image.resize(image, img_size)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image, tf.one_hot(label, depth=80)  # One-hot encode labels (80 classes)

# Load and preprocess the validation dataset
coco_val = load_coco_dataset()
coco_val = coco_val.map(lambda img, lbl: preprocess_data(img, lbl), num_parallel_calls=tf.data.AUTOTUNE)
coco_val = coco_val.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, Model

# Load DenseNet pre-trained on ImageNet
base_model = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Add custom classification head
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)  # Add dropout for regularization
output = layers.Dense(80, activation='softmax')(x)  # 80 classes for COCO dataset

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers to fine-tune the head first
base_model.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
# Train the model
history = model.fit(coco_val, epochs=10, validation_data=coco_val)

# Unfreeze the base model
base_model.trainable = True

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
fine_tune_history = model.fit(coco_val, epochs=5, validation_data=coco_val)

import numpy as np

# Get an example image from the dataset
for img_batch, label_batch in coco_val.take(1):
    img = img_batch[0]  # Take the first image
    true_label = label_batch[0]  # Take the corresponding label

# Expand dimensions and run inference
pred = model.predict(np.expand_dims(img, axis=0))
predicted_class = np.argmax(pred)

print(f"Predicted class: {predicted_class}, True label: {np.argmax(true_label)}")
