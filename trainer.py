from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from DataHandler import download_data_to_local_directory

from tensorflow.python.client import device_lib
import argparse

import hypertune

# helps see which hardware is available for training
# print("Tensorflow is running on the following devices: ")
# print(device_lib.list_local_devices())



def build_model(nbr_classes):

    # base architecture for transfer learning uses pretrained model then removes a
    # layer or 2 and we add our own and train on those layers, train with imagenet
    # we dont want the output layers since we want our own layers so False
    # input size of our NN is of shape original size of network
    base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(229, 229, 3)))

    head_model = base_model.output
    # include custom layers
    head_model = Flatten()(head_model)
    head_model = Dense(512)(head_model)
    head_model = Dropout(0.5)(head_model) # can change parameters for improvement of model learning
    head_model = Dense(nbr_classes, activation="softmax")(head_model)

    # final model to construct full NN based on the provided input/output
    model = Model(inputs=base_model.input, outputs=head_model)

    # dont train(change params) of base model, we want to train other layers above
    for layer in base_model.layers:
        layer.trainable = False

    return model


def build_data_pipelines(batch_size, train_data_path, val_data_path, eval_data_path):

    # img data generator
    train_augmentor = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=25,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
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
        target_size=(229, 229),
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = val_augmentor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(229, 229),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    eval_generator = val_augmentor.flow_from_directory(
        eval_data_path,
        class_mode="categorical",
        target_size=(229, 229),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, eval_generator

# func to get train and val sizes
def get_number_of_imgs_inside_folder(directory):
    totalcount = 0

    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in [".png", ".jpg", ".jpeg"]:
                totalcount = totalcount + 1

    return totalcount


# func to do the training
def train(path_to_data, batch_size, epochs, learning_rate):
    path_train_data = os.path.join(path_to_data, 'training')
    path_val_data = os.path.join(path_to_data, 'validation')
    path_eval_data = os.path.join(path_to_data, 'evaluation')

    # get total number of imgs per dataset
    total_train_imgs = get_number_of_imgs_inside_folder(path_train_data)
    total_val_imgs = get_number_of_imgs_inside_folder(path_val_data)
    total_eval_imgs = get_number_of_imgs_inside_folder(path_eval_data)

    print(total_train_imgs, total_val_imgs,total_eval_imgs)

    train_generator, val_generator, eval_generator = build_data_pipelines(
        batch_size=batch_size,
        train_data_path=path_train_data,
        val_data_path=path_val_data,
        eval_data_path=path_eval_data
    )
    # keys of dict will be names of folders ie bread, soup and values it will have the labels
    classes_dict = train_generator.class_indices

    # create model
    model = build_model(nbr_classes=len(classes_dict.keys()))
    # can use SDG
    optimizer = Adam(lr=learning_rate) #1e-5

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    model.fit_generator(
        train_generator,
        steps_per_epoch=total_train_imgs // batch_size,
        validation_data=val_generator,
        validation_steps=total_val_imgs // batch_size,
        epochs=epochs
    )

    # prediction evaluation
    print("[INFO] Evaluation phase...")

    predictions = model.predict_generator(eval_generator)
    # gives a array of the correct class that was predicted based on highest probability
    predictions_idxs = np.argmax(predictions, axis=1)

    # contains metrics about eval phase, gives precision, accuracy , recall, F1-score and support
    # gives true label int ie if img is of type bread it will be indx 0
    # eval_generator.classes-> ground truth
    my_classification_report = classification_report(eval_generator.classes, predictions_idxs,
                                                     target_names=eval_generator.class_indices.keys())

    # helps to compare predictions of one class to another
    # shows what the model predicted the images to be
    my_confusion_matrix = confusion_matrix(eval_generator.classes, predictions_idxs)

    print("[INFO] Classification report :")
    print(my_classification_report)
    print("[INFO] Confusion matrix :")
    print(my_confusion_matrix)

    # takes images and corresponding labels and coputes a score based on examples
    print("Starting evaluation using model.evaluate_generator")
    # scores returns loss at [0] and accuracy at [1]
    scores = model.evaluate_generator(eval_generator)
    print("Done evaluating!")
    loss = scores[0]
    print(f"loss for hypertune = {loss}")
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='loss', metric_value=loss, global_step=epochs)

if __name__ == "__main__":

    # allow to pass arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket_name", type=str, help="Bucket name on google cloud storage", default= "gcp-food-data-bucket")
    parser.add_argument("--batch_size", type=int, help="Batch size used by deep learning model", default=2)
    
    parser.add_argument("--learning_rate", type=float, help="Learning rate used by deep learning model", default=1e-5)
    
    args = parser.parse_args()
    # python trainer.py --bucket_name "gcp-food-data-bucket" --batch_size 1


    # Can create a small subset folder of the dataset to test if its working properly, say 60,20,20 img split
    # recommended not to store data locally and instead store it in the cloud
    # training will be done locally however data will now be read from a bucket and download from the bucket
    # path_to_data = 'C:/Users/cpere/Downloads/food-dataset/food-11/'
    print("Downloading of data started ...")
    download_data_to_local_directory(args.bucket_name, "./data")
    print("Download finished.")

    path_to_data = './data'
    # added learning rate
    train(path_to_data, args.batch_size, 10, args.learning_rate)

    # python trainer.py --bucket_name "gcp-food-data-bucket" --batch_size 1
