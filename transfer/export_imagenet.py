import os
import sys

# from keras.applications import *
from inception_resnet_v2 import InceptionResnetV2_model
from keras import backend as k
from keras.layers import *
from keras.models import Model
from keras.optimizers import *

# hyper parameters for model
nb_classes = 1  # number of classes
# change based on the shape/structure of your images
img_width, img_height = 299, 299


def export(model_path):
    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
    # base_model = InceptionResnetV2_model(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
    base_model = InceptionResnetV2_model(input_shape=(
        img_width, img_height, 3), weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    # add your top layer block to your base model
    model = Model(base_model.input, predictions)
    print(model.summary())

    model.save(model_path + '/inception_resnet_v2.h5')
    # save model
    # model_json = model.to_json()
    # with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
    #     json_file.write(model_json)


if __name__ == '__main__':
    export(os.getcwd())  # train model

    # release memory
    k.clear_session()
