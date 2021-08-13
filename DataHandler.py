import glob
import shutil
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from decouple import config
from google.cloud import storage

# path_to_credentials = '' # without storage admin role
path_to_credentials = config('PATH_TO_CRED') #with storage admin role
# oceanic-actor-319819-a9b6bb1f2bc0.json owner role only

# 0 index corresponds to 0 that the names of the images start with
food_classes = ['bread', 'dairy_product', 'dessert', 'egg', 'fried_food', 'meat', 'noodle_pasta',
                'rice', 'seafood', 'soup', 'vegetable']

# func that splits data into diff folders, class_id 0-11
# will go through all the images in path_to_data

def split_data_into_class_folders(path_to_data, class_id):

    imgs_paths = glob.glob(path_to_data + '*.jpg')

    for path in imgs_paths:

        basename = os.path.basename(path)
# read first letters of path and put in a new folder that corresponds to one of the classes
        if basename.startswith(str(class_id) + '_'):

            path_to_save = os.path.join(path_to_data, food_classes[class_id])
# verify if path exists
            if not os.path.isdir(path_to_save):
                os.makedirs(path_to_save)

            shutil.move(path, path_to_save)


# func randomly chooses images and shows them
def visualize_some_images(path_to_data):

    imgs_path = []
    labels = []

    # read paths of images in folders that are in folders
    for r, d, f in os.walk(path_to_data):
        for file in f:
            if file.endswith(".jpg"):
                imgs_path.append(os.path.join(r, file))
                labels.append(os.path.basename(r))

    fig = plt.figure()
    # choose and display images
    for i in range(9):
        chosen_index = random.randint(0, len(imgs_path)-1)
        chosen_img = imgs_path[chosen_index]
        chosen_label = labels[chosen_index]

        ax = fig.add_subplot(3, 3, i+1)
        ax.title.set_text(chosen_label) # give it a title of its folder name
        ax.imshow(Image.open(chosen_img))
    
    fig.tight_layout(pad=.05)
    plt.show()


# get idea of width and height
def get_images_sizes(path_to_data):

    imgs_path = []
    widths = []
    heights = []

    for r,d,f in os.walk(path_to_data):
        for file in f:
            if file.endswith(".jpg"):

                img = Image.open(os.path.join(r, file))
                widths.append(img.size[0]) # get width of image
                heights.append(img.size[1]) # get height of image
                img.close()

    mean_width = sum(widths) / len(widths)
    mean_height = sum(heights) / len(heights)
    median_width = np.median(widths)
    median_height = np.median(heights)

    return mean_width, mean_height, median_width, median_height

# func to list different files in bucket
def list_blobs(bucket_name):

    storage_client = storage.Client.from_service_account_json(path_to_credentials)

    blobs = storage_client.list_blobs(bucket_name)

    return blobs

# func that allows to download data from bucket to local directory
def download_data_to_local_directory(bucket_name, local_directory):
    
    storage_client = storage.Client.from_service_account_json(path_to_credentials)
    blobs = storage_client.list_blobs(bucket_name)

    # if local directory doesnt exist, make one
    if not os.path.isdir(local_directory):
        os.makedirs(local_directory)
        
    for blob in blobs:
        
        joined_path = os.path.join(local_directory, blob.name)
        
        # we have a blob of a folder not an image
        if os.path.basename(joined_path) =='':
            if not os.path.isdir(joined_path):
                os.makedirs(joined_path)
        else:
            if not os.path.isfile(joined_path): # check if file doesnt exist
                if not os.path.isdir(os.path.dirname(joined_path)): # check if folders exist in which this file exists
                    os.makedirs(os.path.dirname(joined_path)) # create the folder

                blob.download_to_filename(joined_path)
        
# function uploads data to a bucket
def upload_data_to_bucket(bucket_name, path_to_data, bucket_blob_name):

    storage_client = storage.Client.from_service_account_json(path_to_credentials)
    bucket = storage_client.get_bucket(bucket_name) # obj stores reference to bucket

    blob = bucket.blob(bucket_blob_name)
    blob.upload_from_filename(path_to_data)


if __name__ == '__main__':
    # switches to not rerun code
    split_data_switch = False
    visualize_data_switch = False
    print_insights_switch = False
    list_blobs_switch = False
    download_data_switch = True

    #importing the datasets (no longer will use)
    path_to_train_data = "C:/Users/cpere/Downloads/food-dataset/food-11/training/"
    path_to_val_data = "C:/Users/cpere/Downloads/food-dataset/food-11/validation/"
    path_to_eval_data = "C:/Users/cpere/Downloads/food-dataset/food-11/evaluation/"

    # assign photos to respective newly created folders for each food class
    if split_data_switch:
        for i in range(11):
            split_data_into_class_folders(path_to_train_data, i)
        for i in range(11):
            split_data_into_class_folders(path_to_val_data, i)
        for i in range(11):
            split_data_into_class_folders(path_to_eval_data, i)

    if visualize_data_switch:
        visualize_some_images(path_to_train_data)

    if print_insights_switch:
        mean_width, mean_height, median_width, median_height = get_images_sizes(path_to_train_data)

        print(f"mean width = {mean_width}")
        print(f"mean height = {mean_height}")
        print(f"median width = {median_width}")
        print(f"median width = {median_height}")

    if list_blobs_switch:
        blobs = list_blobs('gcp-food-data-bucket')

        for blob in blobs:
            print(blob.name)

    # note this creates a folder of the data locally, however this isnt a necessary step as we can
    # leave it and access it on gcp
    if download_data_switch:
        download_data_to_local_directory("gcp-food-data-bucket", "./data")
