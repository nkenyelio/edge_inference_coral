import tflite_runtime.interpreter as tflite
import pickle
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load a CIFAR-10 batch
def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    labels = np.array(labels)
    return images, labels

# Load test data
test_images, test_labels = load_cifar10_batch('data/cifar-10-batches-py/test_batch')

# Load class names
with open('data/cifar-10-batches-py/batches.meta', 'rb') as f:
    label_names = pickle.load(f, encoding='bytes')[b'label_names']
label_names = [label.decode('utf-8') for label in label_names]

print(f"Loaded {len(test_images)} test images.")
def resource_cost_function(model_name, latency, memory_usage, power_consumption):
    """
    Calculate resource cost based on latency, memory, and power.
    Lower cost is better.
    """
    weight_latency = 0.4
    weight_memory = 0.3
    weight_power = 0.3

    cost = (weight_latency * latency) + (weight_memory * memory_usage) + (weight_power * power_consumption)
    print(f"[Resource Cost] Model: {model_name}, Cost: {cost:.2f}")
    return cost

MODEL_PATHS = {
    'mobilenet': 'mobilenet_edgetpu.tflite',
    'efficientnet': 'efficientnet_edgetpu.tflite',
    'resnet': 'resnet_edgetpu.tflite'
}

interpreters = {}
for model_name, model_path in MODEL_PATHS.items():
    interpreters[model_name] = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
    interpreters[model_name].allocate_tensors()
def infer_with_model(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare Input
    input_data = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run Inference
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)
    confidence = np.max(output_data)

    return predicted_label, confidence

def dynamic_model_switch(image):
    models = ['mobilenet', 'efficientnet', 'resnet']
    resource_profiles = {
        'mobilenet': {'latency': 50, 'memory': 100, 'power': 5},
        'efficientnet': {'latency': 70, 'memory': 120, 'power': 6},
        'resnet': {'latency': 90, 'memory': 150, 'power': 8}
    }

    best_model = None
    lowest_cost = float('inf')

    for model in models:
        cost = resource_cost_function(
            model_name=model,
            latency=resource_profiles[model]['latency'],
            memory_usage=resource_profiles[model]['memory'],
            power_consumption=resource_profiles[model]['power']
        )
        if cost < lowest_cost:
            lowest_cost = cost
            best_model = model

    print(f"Selected Model: {best_model}")
    prediction, confidence = infer_with_model(interpreters[best_model], image)
    print(f"Prediction: {class_names[prediction]}, Confidence: {confidence:.2f}")
    return best_model, prediction, confidence


num_samples = 10
for i in range(num_samples):
    idx = random.randint(0, len(test_images) - 1)
    test_image = test_images[idx]
    true_label = label_names[test_labels[idx]]

    print(f"\n[Sample {i+1}] True Label: {true_label}")
    model, prediction, confidence = dynamic_model_switch(test_image)

resource_data = {
    'model': ['mobilenet', 'efficientnet', 'resnet'],
    'latency': [50, 70, 90],
    'memory': [100, 120, 150],
    'power': [5, 6, 8],
    'cost': [55, 72, 89]
}

df = pd.DataFrame(resource_data)
sns.barplot(data=df, x='model', y='cost', palette='muted')
plt.title('Resource Cost Across Models')
plt.ylabel('Resource Cost')
plt.xlabel('Model')
plt.show()
