#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
FACE RECOGNITION SERVER - a RESTful API for face_recognition on Linux servers using Python Flask.
"""

import os, sys
import json

import pickle
from datetime import datetime

import cv2
import imutils
from PIL import Image, ImageDraw

#from sklearn import neighbors

import face_recognition as fr
# from face_recognition import load_image_file
# from face_recognition.face_recognition_cli import image_files_in_folder
# from face_util import compare_faces, face_rec

from flask import Flask, request, render_template


app = Flask(__name__)

UPLOAD_FOLDER = "./static/img"

MODEL_PATH = "./models/trained_knn_model_600_neighbors.clf"

# IMAGE_PATH = "./img_sample_test/"
OUTPUT_FOLDER = "./static/img_op"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.45):
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
    X_img = fr.load_image_file(X_img_path)
    X_face_locations = fr.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = fr.face_encodings(X_img, known_face_locations=X_face_locations)

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
    image = fr.load_image_file(image_path)
    height = image[0]
    width = image[1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = fr.face_locations(image)
    #face_encodings = face_recognition.face_encodings(image, face_locations)

    print("Found {} face(s) in this photograph.".format(len(face_locations)))
    face_count = 0

    face_names = dict()

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
        #face_image = cv2.resize(face_image, (200, 200))
        face_image = imutils.resize(face_image, width=200)
        pil_image = Image.fromarray(face_image)
        #pil_image = imutils.resize(pil_image, width=200)
        #pil_image.show()

        # face_image_found = []

        # # Remove face image if already exist!
        # if os.path.isdir(output_folder_path):
        #     for image_f in os.listdir(output_folder_path):
        #         full_file_path = os.path.join(output_folder_path, image_f)
        #         #os.chmod(full_file_path, stat.S_IWRITE)
        #         os.chmod(full_file_path, 0o777)
        #         os.remove(full_file_path)
        date_time = datetime.now().strftime("%d_%m_%y|%H:%M:%S")
        FaceFileName = os.path.join(output_folder_path, date_time + "_" + str(face_count) + ".jpg")

        # save fresh Image
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
            date = datetime.now().strftime("%d_%m_%y|%H:%M:%S")
            new_name_file = os.path.join(output_folder_path, name + "_" + str(face_count) + date + ".jpg")
            os.rename(FaceFileName, new_name_file)

            name = name.replace("_", " ")
            name_resp_data = {new_name_file.split('/')[-1]: name}
            face_names.update(name_resp_data)

            # logic to return predicted face names list
            # dict to get filenames
            # if os.path.isdir(output_folder_path):
            #     for image_file in os.listdir(output_folder_path):
            # name_resp_data = {new_name_file: name}
            # face_names.append(name_resp_data)

        # Display results overlaid on an image
        #show_prediction_labels_on_image(new_name_file, predictions)

    # jsonify dict object
    return json.dumps(face_names)



# @app.route('/face_match', methods=['POST'])
# def face_match():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if ('file1' in request.files) and ('file2' in request.files):        
#             file1 = request.files.get('file1')
#             file2 = request.files.get('file2')
#             ret = compare_faces(file1, file2)
#             resp_data = {"match": bool(ret)} # convert numpy._bool to bool for json.dumps
#             return json.dumps(resp_data) 


@app.route('/', methods=['GET', 'POST'])
def face_recognition():
    #face_names = []
    if request.method == 'POST':
        # check if the post request has the file part
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)

            print("Looking for faces in {}".format(image_location.split('/')[-1]))
            print(image_location)
            #print(image_location.split('/')[-1])

            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            # predictions = predict(image_location, model_path=MODEL_PATH)
            # # Print results on the console
            # for name, (top, right, bottom, left) in predictions:
            #     print("- Found {} at ({}, {})".format(name, left, top))
            
                #name = name.replace("_", " ")
                #resp_data = {'name': name}
                #face_name.append(resp_data)

            # Remove face image if already exist!
            if os.path.isdir(OUTPUT_FOLDER):
                for image_f in os.listdir(OUTPUT_FOLDER):
                    full_file_path = os.path.join(OUTPUT_FOLDER, image_f)
                    #os.chmod(full_file_path, stat.S_IWRITE)
                    os.chmod(full_file_path, 0o777)
                    os.remove(full_file_path)
            
            # function to detect faces from image and save it as a face image file with predicted label
            face_file_names = save_predicted_face(image_location, OUTPUT_FOLDER)
            face_file_names = json.loads(face_file_names)

            return render_template('index.html', prediction = face_file_names, image_loc = image_location.split('/')[-1])
    return render_template('index.html', prediction = 'face_name', image_loc = None)
    #return 'OK'


if __name__ == "__main__":
    # When debug = True, code is reloaded on the fly while saved
    app.run(host='127.0.0.1', port='5001', debug=True)