import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import Model, layers
import numpy as np
import json

# Path to COCO dataset and annotations
coco_dir = "coco/val2017/"
annotation_file = "coco/annotations_trainval2017/annotations/instances_val2017.json"

# Load annotations
with open(annotation_file, "r") as f:
    coco_annotations = json.load(f)

# Map COCO category IDs to zero-based indices
category_mapping = {cat["id"]: i for i, cat in enumerate(coco_annotations["categories"])}

# Create labels for images
image_labels = {}
for annotation in coco_annotations["annotations"]:
    image_id = annotation["image_id"]
    category_id = category_mapping[annotation["category_id"]]

    if image_id not in image_labels:
        image_labels[image_id] = [0] * 80
    image_labels[image_id][category_id] = 1

# Convert labels to NumPy array
labels_array = np.array(list(image_labels.values()))

# Load and preprocess COCO dataset
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    coco_dir,
    label_mode=None,  # Images only
    image_size=(299, 299),
    batch_size=32
)
val_dataset = val_dataset.map(lambda x: x / 255.0)  # Normalize images

# Combine dataset with labels
labels_dataset = tf.data.Dataset.from_tensor_slices(labels_array)
val_dataset_with_labels = tf.data.Dataset.zip((val_dataset, labels_dataset))

# Load pre-trained Xception model
base_model = Xception(input_shape=(299, 299, 3), include_top=False, weights="imagenet")
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(80, activation="softmax")(x)  # 80 classes for COCO
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(val_dataset_with_labels, epochs=5)


for images, labels in val_dataset_with_labels.take(1):
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

