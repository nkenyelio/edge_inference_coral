import argparse
import time

from PIL import Image
from PIL import ImageDraw

import detect
import tflite_runtime.interpreter as tflite


def load_labels(path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')


def resource_cost_function(latency, memory_usage, power_consumption):
    return (0.4 * latency) + (0.3 * memory_usage) + (0.3 * power_consumption)


# import tflite_runtime.interpreter as tflite


def load_interpreter(model_path):
    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
    interpreter.allocate_tensors()
    return interpreter


# Load compiled models for inference
models = {
    'Xception': load_interpreter('xception_coco_edgetpu.tflite'),
    'DenseNet': load_interpreter('densenet_coco_edgetpu.tflite'),
    'NASNetMobile': load_interpreter('nasnetmobile_coco_edgetpu.tflite'),
    'VGG16': load_interpreter('vgg16_coco_edgetpu.tflite'),
    'InceptionResNetV2': load_interpreter('inceptionresnet_coco_edgetpu.tflite')
}
resource_profiles = {
    'Xception': {'latency': 60, 'memory': 150, 'power': 8},
    'DenseNet': {'latency': 70, 'memory': 180, 'power': 9},
    'NASNetMobile': {'latency': 80, 'memory': 120, 'power': 7},
    'VGG16': {'latency': 90, 'memory': 140, 'power': 10},
    'InceptionResNetV2': {'latency': 100, 'memory': 160, 'power': 12}
}


def dynamic_model_switch(label_fnc, image_input, output_img, count=5,threshold=0.4):
    best_model = None
    lowest_cost = float('inf')
    label = label_fnc

    for model_name, resources in resource_profiles.items():
        cost = resource_cost_function(resources['latency'], resources['memory'], resources['power'])
        if cost < lowest_cost:
            lowest_cost = cost
            best_model = model_name

    interpreter = models[best_model]
    #input_details = interpreter.get_input_details()
    #output_details = interpreter.get_output_details()
    image = Image.open(image_input)
    scale = detect.set_input(interpreter, image.size,
                           lambda size: image.resize(size, Image.ANTIALIAS))


    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes',
          'loading the model into Edge TPU memory.')
    for _ in range(count):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_output(interpreter, threshold, scale)
        print('%.2f ms' % (inference_time * 1000))

    print('-------RESULTS--------')
    if not objs:
        print('No objects detected')

    for obj in objs:
        print(label.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)

    # if output_img:
    image = image.convert('RGB')
    draw_objects(ImageDraw.Draw(image), objs, label)
    image.save(output_img)
    image.show()

    # image = np.expand_dims(image, axis=0).astype(np.float32)
    # interpreter.set_tensor(input_details[0]['index'], image)
    # interpreter.invoke()
    # output = interpreter.get_tensor(output_details[0]['index'])

    # prediction = np.argmax(output)
    # confidence = np.max(output)

    #print(f"Model: {best_model}, Inference: {inference_time}, Confidence: {confidence:.2f}")
    print(f"Model: {best_model}, Inference: {inference_time}")


import cv2

label = 'coco_labels.txt'
image_input = 'images/grace_hopper.bmp'
output_rest = 'images/grace_hopper_processed_'
# Assuming 'image' is an image from the COCO dataset
# image = cv2.imread('coco/val2017/000000000139.jpg')
dynamic_model_switch(label, image_input, output_rest )
