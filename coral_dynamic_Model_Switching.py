import tensorflow as tf
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

# Resource profiles for each model (example values)
resource_profiles = {
    'mobilenet': {'latency': 50, 'memory': 100, 'power': 5},
    'resnet': {'latency': 70, 'memory': 120, 'power': 7},
    'densenet': {'latency': 60, 'memory': 110, 'power': 6},
    'vgg': {'latency': 90, 'memory': 150, 'power': 10},
    'nasnet': {'latency': 85, 'memory': 130, 'power': 8},

}

def resource_cost_function(latency, memory_usage, power_consumption):
    return (0.4 * latency) + (0.3 * memory_usage) + (0.3 * power_consumption)

# Load Edge TPU models
def load_tflite_model(model_name):
    interpreter = tf.lite.Interpreter(model_path=f'{model_name}_edge_cifar10_edgetpu.tflite')
    interpreter.allocate_tensors()
    return interpreter

#models = {name: load_tflite_model(name) for name in resource_profiles.keys()}

models = {
    'mobilenet': load_tflite_model('mobilenet_edge_cifar10_edgetpu.tflite'),
    'resnet': load_tflite_model('resnet_edge_cifar10_edgetpu.tflite'),
    'densenet': load_tflite_model('densenet_edge_cifar10_edgetpu.tflite')
}

# Dynamic model selection based on resource cost
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

    # Prepare image for inference
    image = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.argmax(output)
    print(f"Model: {best_model}, Prediction: {prediction}")

# Run inference dynamically on a CIFAR-10 test image
image = test_images[0]  # Use the first test image for this example
dynamic_model_switch(image)
