import os
import sys

# from keras.applications import *
from inception_resnet_v2 import InceptionResnetV2_model
from keras import backend as k
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import *
from keras.models import Model
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# hyper parameters for model
nb_classes = 15  # number of classes
# change based on the shape/structure of your images
img_width, img_height = 299, 299
# try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
batch_size = 32
nb_epoch = 200  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
# how aggressive will be the data augmentation/transformation
transformation_ratio = .05


def export(model_path):
    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
    # base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
    base_model = InceptionResnetV2_model(input_shape=(
        img_width, img_height, 3), include_top=False)

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
    model_dir = './'

    os.makedirs(model_dir, exist_ok=True)

    export(model_dir)  # train model

    # release memory
    k.clear_session()
