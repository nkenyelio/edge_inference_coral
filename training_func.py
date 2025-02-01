import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, DenseNet121, VGG16, Xception, NASNetMobile, InceptionV3
import os
import numpy as np
import pickle

# Load CIFAR-10 Data
def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    labels = np.array(labels)
    return images, labels

# Load CIFAR-10 data (from 'cifar-10-batches-py')
test_images, test_labels = load_cifar10_batch('data/cifar-10-batches-py/test_batch')

# Model Training function
def create_and_train_model(model_name):
    if model_name == 'mobilenet':
        base_model = MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights=None)
    elif model_name == 'resnet':
        base_model = ResNet50(input_shape=(32, 32, 3), include_top=False, weights=None)
    elif model_name == 'densenet':
        base_model = DenseNet121(input_shape=(32, 32, 3), include_top=False, weights=None)
    elif model_name == 'vgg':
        base_model = VGG16(input_shape=(32, 32, 3), include_top=False, weights=None)
    elif model_name == 'xception':
        base_model = Xception(input_shape=(32, 32, 3), include_top=False, weights=None)
    elif model_name == 'nasnet':
        base_model = NASNetMobile(input_shape=(32, 32, 3), include_top=False, weights=None)
    elif model_name == 'inception':
        base_model = InceptionV3(input_shape=(32, 32, 3), include_top=False, weights=None)
    else:
        raise ValueError("Invalid model name")

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(test_images, test_labels, epochs=5, validation_split=0.2)
    model.save(f'{model_name}_cifar10_saved_model')
    print(f"{model_name} model trained and saved.")

# Train and Save Models (10 models)
for model_name in ['mobilenet', 'resnet', 'densenet', 'vgg', 'xception', 'nasnet', 'inception']:
    create_and_train_model(model_name)
