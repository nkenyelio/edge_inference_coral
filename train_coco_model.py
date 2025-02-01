import tensorflow as tf
from tensorflow.keras.applications import Xception, DenseNet121, NASNetMobile, VGG16, InceptionResNetV2, MobileNetV2, ResNet50
from tensorflow.keras import layers, models
import os

def create_and_train_model(model_name, input_shape=(224, 224, 3), num_classes=80):
    if model_name == 'mobilenet':
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    elif model_name == 'resnet':
        base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categoxceptionrical_crossentropy',
                  metrics=['accuracy'])

    model.save(f'{model_name}_coco_saved_model')
    print(f'{model_name} model saved.')

# Train models (this part can be time-consuming)
create_and_train_model('mobilenet')
create_and_train_model('resnet')
#create_and_train_model('nasnetmobile')
#create_and_train_model('vgg16')
#create_and_train_model('inceptionresnet')
#create_and_train_model('yolo')
