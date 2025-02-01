import tflite_runtime.interpreter as tflite

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'

try:
    interpreter = tflite.Interpreter(
        model_path='cifar10_model_edgetpu.tflite',
        experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)]
    )
    print("Edge TPU Delegate Loaded Successfully!")
except Exception as e:
    print(f"Failed to load Edge TPU delegate: {e}")
