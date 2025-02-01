import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras import layers, models
import pickle
import numpy as np
import tflite_runtime.interpreter as tflite

def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    labels = np.array(labels)
    return images, labels

# Load Test Data
test_images, test_labels = load_cifar10_batch('data/cifar-10-batches-py/test_batch')

# Load Class Names
with open('data/cifar-10-batches-py/batches.meta', 'rb') as f:
    label_names = pickle.load(f, encoding='bytes')[b'label_names']
label_names = [label.decode('utf-8') for label in label_names]

print(f"Loaded {len(test_images)} CIFAR-10 test images.")

def resource_cost_function(latency, memory_usage, power_consumption):
    return (0.4 * latency) + (0.3 * memory_usage) + (0.3 * power_consumption)


def load_interpreter(model_path):
    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
    interpreter.allocate_tensors()
    return interpreter

resource_profiles = {
    'mobilenet': {'latency': 50, 'memory': 100, 'power': 5},
    'efficientnet': {'latency': 70, 'memory': 120, 'power': 7},
    'resnet': {'latency': 90, 'memory': 150, 'power': 10}
}

models = {
    'mobilenet': load_interpreter('mobilenet_cifar10_edgetpu.tflite'),
    'efficientnet': load_interpreter('cifar10_model_edgetpu.tflite'),
    'resnet': load_interpreter('resnet_cifar10_edgetpu.tflite')
}

def dynamic_model_switch(image):
    best_model = None
    lowest_cost = float('inf')

    for model_name, resources in resource_profiles.items():
        cost = resource_cost_function(resources['latency'], resources['memory'], resources['power'])
        if cost < lowest_cost:
            lowest_cost = cost
            best_model = model_name

    interpreter = models[best_model]
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.argmax(output)
    confidence = np.max(output)

    print(f"Model: {best_model}, Prediction: {label_names[prediction]}, Confidence: {confidence:.2f}")
for i in range(10):
    dynamic_model_switch(test_images[i])
