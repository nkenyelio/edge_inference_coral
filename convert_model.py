
import tensorflow as tf

def convert_to_tflite(saved_model_path, tflite_model_path):
    # Convert saved model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()

    # Save TFLite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Converted {saved_model_path} to {tflite_model_path}")

# Convert models to TFLite
for model_name in ['mobilenet', 'resnet', 'densenet', 'vgg', 'nasnet']:
    convert_to_tflite(f'{model_name}_edge_cifar10_saved_model', f'{model_name}_edge_cifar10.tflite')
