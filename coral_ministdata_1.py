import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate
import time

# Load and preprocess the MNIST dataset
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    return x_train, y_train, x_test, y_test

# Train a simple model (extend to multiple models as needed)
def train_mnist_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    x_train, y_train, _, _ = load_mnist_data()
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    model.save("mnist_model")
    return model

# Convert and quantize the model to TFLite format
def quantize_model(saved_model_dir, output_tflite_file):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.representative_dataset = lambda: (x[None, ...] for x in load_mnist_data()[0][:1000])
    tflite_model = converter.convert()
    with open(output_tflite_file, "wb") as f:
        f.write(tflite_model)
    return output_tflite_file

# Compile the quantized model for the Edge TPU
def compile_model_for_tpu(tflite_model_file):
    import os
    os.system(f"edgetpu_compiler {tflite_model_file}")

# Resource consumption function
def resource_consumption(latency, memory, flops, tpu_usage, accuracy):
    weights = {"latency": 0.4, "memory": 0.2, "flops": 0.2, "tpu_usage": 0.1, "accuracy": 0.1}
    return (
        weights["latency"] * latency +
        weights["memory"] * memory +
        weights["flops"] * flops +
        weights["tpu_usage"] * tpu_usage -
        weights["accuracy"] * accuracy  # Maximize accuracy
    )

# Perform inference on a single image and log metrics
def infer_on_model(model_path, input_image):
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate("libedgetpu.so.1")])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input
    interpreter.set_tensor(input_details[0]["index"], input_image)

    # Measure latency
    start_time = time.time()
    interpreter.invoke()
    latency = time.time() - start_time

    # Extract output and simulate accuracy and resource metrics
    output = interpreter.get_tensor(output_details[0]["index"])
    accuracy = np.random.uniform(0.8, 0.99)  # Simulated accuracy
    memory = np.random.uniform(50, 150)  # Simulated memory in MB
    flops = np.random.uniform(1e6, 1e8)  # Simulated FLOPs
    tpu_usage = np.random.uniform(40, 80)  # Simulated TPU usage in percentage

    return latency, memory, flops, tpu_usage, accuracy

# Dynamic model switching
def dynamic_model_switch(input_image, application_complexity=0.5):
    model_paths = {
        "model1": "mnist_model_edgetpu.tflite",
        # Add more models as needed
    }
    best_model = None
    best_score = float("inf")

    for model_name, model_path in model_paths.items():
        latency, memory, flops, tpu_usage, accuracy = infer_on_model(model_path, input_image)
        score = resource_consumption(latency, memory, flops, tpu_usage, accuracy)
        adjusted_score = score * (1 + application_complexity)
        print(f"Model: {model_name}, Adjusted Score: {adjusted_score:.4f}, Latency: {latency:.4f}s")

        if adjusted_score < best_score:
            best_score = adjusted_score
            best_model = model_name

    return best_model

# Main function
if __name__ == "__main__":
    # Train and quantize models
    #train_mnist_model()
    quantize_model("mnist_model", "mnist_model.tflite")
    compile_model_for_tpu("mnist_model.tflite")
    
    # Perform dynamic model switching
    input_image = load_mnist_data()[2][0][None, ...]  # First test image
    best_model = dynamic_model_switch(input_image)
    print(f"Best Model for the given input: {best_model}")
