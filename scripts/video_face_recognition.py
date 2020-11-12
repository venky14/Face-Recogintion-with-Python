"""
Face Detection and Recognition From Video Stream Data
"""

import sys
import os
import pickle
#import dlib
import cv2
from datetime import datetime
import imutils
from sklearn import neighbors
from PIL import Image, ImageDraw
# import urllib.request
# from urllib.request import urlopen
import face_recognition
#from face_recognition.face_recognition_cli import image_files_in_folder

MODEL_PATH = "./models/trained_knn_model_600_neighbors.clf"

#VIDEO_PATH = "./vid_sample_test/Narendra_Modi_clip.mp4"
VIDEO_PATH = "./vid_sample_test/Modi_Akshay.mp4"


def video_stream_capture():
	input_stream = cv2.VideoCapture(VIDEO_PATH)

	channel_name = "TV_NEWS"
	channel_date_time = datetime.now().strftime("%d_%m_%y")
	output_folder_path = "./VS_FR/" + channel_date_time + "/" + channel_name + '_' + channel_date_time
	
	# Ensure output directory exists
	if not os.path.isdir(output_folder_path):
		os.makedirs(output_folder_path, 0o777)
		#print(output_folder_path)
		#exit()


	frame_number = 0
	face_count = len(os.listdir(output_folder_path))
	print(face_count)
	print("Video Processing Started")
	
	while True:
		# Grab a single frame of video
		ret, frame = input_stream.read()
		frame_number += 1

		#capture only 20th frame from stream
		if frame_number == 10:
			if ret == True:
				time_text_on = datetime.now().strftime("%d_%m_%Y||%H:%M:%S")
				log_name_text = "TV NEWS Streaming is ON||" + time_text_on
				text_file_folder = "./VS_FR/" + channel_date_time + "/"
				text_file = open(os.path.join(text_file_folder, channel_name + '_' + channel_date_time + '_log.txt'), 'a+')
				text_file.write("{}||".format(log_name_text))
				text_file.write("\n")
				text_file.close()
				#cv2.imshow('captured_frame', frame)
				# Resize the frame for faster processing
				#frame = imutils.resize(frame, width=600)
				# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
				rgb_frame = frame[:, :, ::-1]
				n_sample, h, w = rgb_frame.shape
				height = rgb_frame[0]
				width = rgb_frame[1]

				# Find all the faces and face enqcodings in the frame of video
				face_locations = face_recognition.face_locations(rgb_frame)
				face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

				#print("I found {} face(s) in this photograph.".format(len(face_locations)))

				#face = []
				

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
					face_image = rgb_frame[top:bottom, left:right]
					face_image = imutils.resize(face_image, width=200)
					pil_image = Image.fromarray(face_image)
					#pil_image = imutils.resize(pil_image, width=200)
					#pil_image.show()

					face_image_found = []

					data_time = datetime.now().strftime("%d_%m_%y|%H:%M:%S")
					FaceFileName = os.path.join(output_folder_path, data_time + "_" + str(face_count) + ".jpg")
					pil_image.save(FaceFileName)
					face_image_found.append(FaceFileName)
					#print(FaceFileName)
					#print(face_image_found)

					for image_file in face_image_found:
						#print(len(image_file))
						full_file_path = image_file
						os.chmod(full_file_path, 0o777)

						# Load Face Recognition Model
						fr_model_path = MODEL_PATH
						
						with open(fr_model_path, 'rb') as f:
							knn_clf = pickle.load(f)

						# Predict all people in the image frame using a trained classifier model
						predictions = predict(full_file_path, knn_clf=knn_clf, distance_threshold=0.50)
						#print(predictions)

						for name, (top, right, bottom, left), rec in predictions:
							# Replace name with predicted label
							new_name_file = os.path.join(output_folder_path, name + "_" + str(face_count) + ".jpg")
							os.rename(full_file_path, new_name_file)
							data_time_text = datetime.now().strftime("%d_%m_%Y||%H:%M:%S")
							new_name_text = os.path.join(data_time_text + "||" + "TV_NEWS" + "||" + name + "||" + name + "_" + str(face_count) + ".jpg")
							
							#print("- Found {} at ({}, {}, {}, {})".format(name, top, right, bottom, left))
							#print("successfully renamed {} to {}".format(image_file, new_name))
							text_file_folder = "./VS_FR/" + channel_date_time + "/"
							text_file = open(os.path.join(text_file_folder, channel_name + '_' + channel_date_time + '.txt'), 'a+')
							text_file.write("{}||".format(new_name_text))
							text_file.write("\n")
							text_file.close()
							#output_file = open(os.path.join(text_file_folder, channel_name + '_' + channel_date_time + '.txt'), 'r')
							#print(output_file.read())

			else:
				time_text_off = datetime.now().strftime("%d_%m_%Y||%H:%M:%S")
				log_name_text = "TV NEWS Streaming is OFF||" + time_text_off
				text_file_folder = "./VS_FR/" + channel_date_time + "/"
				text_file = open(os.path.join(text_file_folder, channel_name + '_' + channel_date_time + '_log.txt'), 'a+')
				text_file.write("{}||".format(log_name_text))
				text_file.write("\n")
				text_file.close()
				print(log_name_text)
				sys.exit(0)
				#video_stream_capture() #recursive
				#continue
			frame_number = 0


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
## Prediction - Face Recognition
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=None):
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
        raise Exception(
            "Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)
    #X_face_locations = face_recognition.face_locations(X_face_locations)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) != 1:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(
        X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    #print(closest_distances)

    are_matches = [closest_distances[0][i][0] <=
                    distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc, rec) if rec else ("unknown_person", loc, rec) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


if __name__ == "__main__":

	video_stream_capture()