from tensorflow.keras.preprocessing import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorlfow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam


def build_model(num_classes):

    # base architecture for transfer learning uses pretrained model then removes a
    # layer or 2 and we add our own and train on those layers, train with imagenet
    # we dont want the output layers since we want our own layers so False
    # input size of our NN is of shape original size of network
    base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(229, 229, 3)))

    head_model = base_model.output
    # include custom layers
    head_model = Flatten()(head_model)
    head_model = Dense(512)(head_model)
    head_model = Dropout(.5)(head_model) # can change paramaters for improvement of model learning
    head_model = Dense(num_classes, activation="softmax")(head_model)

    # final model to construct full NN based on the provided input/output
    model = Model(inputs=base_model.input, outputs=head_model)

    # dont train(change params) of base model, we want to train other layers above
    for layer in base_model.layers:
        layer.trainable = False

    return model

# func to create data pipelines, takes our images and applies some transformations to them and output new augmented data
def build_data_pipelines(batch_size,train_data_path, val_data_path, eval_data_path):

    #img data generator
    train_augmentor = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=25,
        zoom_range=.15,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_augmentor = ImageDataGenerator(
        rescale=1. / 255
    )

    # generators will take new data from augementors
    train_generator = train_augmentor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(229,299),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = val_augmentor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(229, 299),
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )

    eval_generator = val_augmentor.flow_from_directory(
        eval_data_path,
        class_mode="categorical",
        target_size=(229, 299),
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, eval_generator
