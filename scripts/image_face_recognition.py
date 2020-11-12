#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
This is an script of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of img1
and two images of img2, The result would be 'img2'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

- Call 'predict' and pass in your trained model to recognize the people in an unknown image.

"""

import os
import pickle
from datetime import datetime

from sklearn import neighbors

from PIL import Image, ImageDraw
import imutils

import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

MODEL_PATH = "./models/trained_knn_model_600_neighbors.clf"

IMAGE_PATH = "./img_sample_test/"
output_folder_path = "./img_sample_test_op/"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.50):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
            of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()



def save_predicted_face(image_path, output_folder_path):
    """
    Function to detect faces from image and save it as a face image file with predicted label

    Arguments:
        image_path {[string]} -- [description]
        output_folder_path {[string]} -- [description]
    """
    image = face_recognition.load_image_file(image_path)
    height = image[0]
    width = image[1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(image)
    #face_encodings = face_recognition.face_encodings(image, face_locations)

    print("Found {} face(s) in this photograph.".format(len(face_locations)))
    face_count = 0

    for face_location in face_locations:
        face_count += 1
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        # print(top)
        # print(right)
        # print(bottom)
        # print(left)

        top = top if (top - 25) < 0 else (top - 25)
        right = right if (right + 25) < width.all() else (right + 25)
        bottom = bottom if (bottom + 25) < height.all() else (bottom + 25)
        left = left if (left - 25) < 0 else (left - 25)
        # print(top)
        # print(right)
        # print(bottom)
        # print(left)
        #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # Access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        face_image = imutils.resize(face_image, width=200)
        pil_image = Image.fromarray(face_image)
        #pil_image = imutils.resize(pil_image, width=200)
        #pil_image.show()

        # face_image_found = []

        data_time = datetime.now().strftime("%d_%m_%y|%H:%M:%S")
        FaceFileName = os.path.join(output_folder_path, data_time + "_" + str(face_count) + ".jpg")
        pil_image.save(FaceFileName)
        #face_image_found.append(FaceFileName)
        #print(FaceFileName)
        #print(face_image_found)

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(FaceFileName, model_path=MODEL_PATH)

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            # Replace name with predicted label
            new_name_file = os.path.join(output_folder_path, name + "_" + str(face_count) + ".jpg")
            os.rename(FaceFileName, new_name_file)

        # Display results overlaid on an image
        #show_prediction_labels_on_image(new_name_file, predictions)


if __name__ == "__main__":

    # STEPS: Using the trained classifier, make predictions for unknown images
    
    for image_file in os.listdir(IMAGE_PATH):
        full_file_path = os.path.join(IMAGE_PATH, image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path=MODEL_PATH)

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        show_prediction_labels_on_image(full_file_path, predictions)

        # -------------------------------------------------------------------------------- #
        # function to detect faces from image and save it as a face image file with predicted label
        save_predicted_face(full_file_path, output_folder_path)