import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import json
import os

# Configuration
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 10
NUM_CLASSES = 80  # COCO has 80 classes
COCO_DIR = 'coco2017'

def preprocess_example(example):
    """Preprocess a single example from COCO dataset"""
    # Extract image and labels from the example dictionary
    image = example['image']
    labels = tf.cast(example['objects']['label'], tf.float32)

    # Create one-hot encoded labels
    label_vector = tf.zeros([NUM_CLASSES])
    label_vector = tf.tensor_scatter_nd_update(
        label_vector,
        tf.expand_dims(tf.cast(labels, tf.int32), 1),
        tf.ones_like(labels, dtype=tf.float32)
    )

    # Preprocess image
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.densenet.preprocess_input(image)

    return image, label_vector

def create_model():
    """Create DenseNet model with COCO specifications"""
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMAGE_SIZE, 3)
    )

    # Add custom layers for COCO classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def load_coco_dataset():
    """Load and prepare COCO dataset"""
    # Load COCO 2017
    print("Loading datasets...")
    train_dataset, validation_dataset = tfds.load(
        'coco/2017',
        split=['train', 'validation'],
        data_dir=COCO_DIR,
        with_info=False
    )

    # Prepare datasets
    train_dataset = (
        train_dataset
        .map(preprocess_example, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_dataset = (
        validation_dataset
        .map(preprocess_example, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, validation_dataset

def train_model(model, train_dataset, validation_dataset):
    """Train the model"""
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Add callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        )
    ]

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset,
        callbacks=callbacks
    )

    return history

def run_inference(model, validation_dataset):
    """Run inference on validation dataset"""
    # Make predictions
    predictions = []
    ground_truth = []

    for images, labels in validation_dataset:
        batch_predictions = model.predict(images)
        predictions.extend(batch_predictions)
        ground_truth.extend(labels.numpy())

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Calculate metrics
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()

    precision.update_state(ground_truth, predictions)
    recall.update_state(ground_truth, predictions)

    results = {
        'precision': float(precision.result().numpy()),
        'recall': float(recall.result().numpy()),
        'f1_score': 2 * (precision.result() * recall.result()) / (precision.result() + recall.result())
    }

    return results

def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)

    # Load dataset
    print("Loading COCO dataset...")
    train_dataset, validation_dataset = load_coco_dataset()

    # Create and train model
    print("Creating model...")
    model = create_model()

    print("Training model...")
    history = train_model(model, train_dataset, validation_dataset)

    # Save training history
    with open('output/training_history.json', 'w') as f:
        json.dump(history.history, f)

    # Run inference
    print("Running inference on validation set...")
    results = run_inference(model, validation_dataset)

    # Save results
    with open('output/inference_results.json', 'w') as f:
        json.dump(results, f)

    print("Results:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main()
