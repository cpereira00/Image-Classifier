from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
import cv2


food_classes = ['bread', 'dairy_product', 'dessert', 'egg', 'fried_food', 'meat', 'noodle_pasta',
                'rice', 'seafood', 'soup', 'vegetable or fruit']

# creating a[[, images will be sent to static directory
app = Flask(__name__, static_url_path='/static')

# uploads images to file and only accepts jpg or
app.config["IMAGE_UPLOADS"] = './static'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]

# loading prediction model I created
food_prediction_model = tf.keras.models.load_model('./deeplearning_model')

# func that makes sure file name is accepted and allowed
def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1] #image.jpg

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

# func that defines what page user sees
@app.route("/", methods=["GET", "POST"])

# func to upload images to website
def upload_image():
    # verify if request is a post and then if there are files attached
    if request.method == "POST":

        if request.files:

            image = request.files["image"]
            # if they upload an empty image
            if image.filename == "":

                return redirect(request.url)
            # secure file name
            if allowed_image(image.filename):

                filename = secure_filename(image.filename)

                #upload image to that static folder
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                # return to new page
                return redirect(f'/showing-image/{filename}')

            # if file image name is not allowed stay on same url
            else:
                return redirect(request.url)
            
    return render_template("upload_images.html")


# new page
@app.route("/showing-image/<image_name>", methods=["GET", "POST"])
# new page for the prediction
def showing_image(image_name):

    if request.method == "POST":

        image_path = os.path.join(app.config["IMAGE_UPLOADS"], image_name)
        # image preprocessing
        image = cv2.imread(image_path)  # BGR
        img = image.copy()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (229, 229))
        image = image.astype("float32")
        image = image / 255.0
        np_image = np.expand_dims(image, axis=0)  # (229,229,3) --> (1,229,229,3) INPUT TO NN

        # predictions
        predictions = food_prediction_model(np_image)

        # return the highest prob of the predictions therefore the "right" prediction
        predicted_class_idx = np.argmax(predictions)  # [0.1, 0.5, 0.3] --> 1
        probability = np.max(predictions)
        predicted_class = food_classes[predicted_class_idx]
        
        return render_template("prediction_result.html", image_name=image_name, predicted_class=predicted_class, probability=probability)

    return render_template("showing_image.html", value=image_name)


if __name__ == '__main__':
    # host 0.0.0.0 PORT 8080
    app.run(debug=True, host='127.0.0.1', port=int(os.environ.get('PORT', 5000)))

