from keras.applications.nasnet import NASNetMobile
from keras.models import Model
from keras.layers import Concatenate, Dense, Input, Dropout, Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D

def NASNet_mobile():
    """Creates a pretrained NASNet mobile, adapted from
    https://www.kaggle.com/CVxTz/cnn-starter-nasnet-mobile-0-9709-lb

    Returns:
        Keras model: NASNet mobile model
    """

    inputs = Input((96, 96, 3))
    base_model = NASNetMobile(include_top=False, input_tensor=inputs, weights='imagenet')
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="out_")(out)
    model = Model(inputs, out)

    return model