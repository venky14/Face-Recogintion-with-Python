import os
import os.path
import pickle
from datetime import datetime
from PIL import Image
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

#image_dir = "/home/kaal/E Paper/Full Article/01/The_Kalinga_Chronicle/THEKALINGACHRONICLE_Bhubaneswar_2019_02_01_8.jpg"
#image_dir = "/home/kaal/face_recognition/examples/knn_examples/train/aamir_khan"

# for image_file in os.listdir(image_dir):
#     img_file_path = os.path.join(image_dir, image_file)
#     image = face_recognition.load_image_file(img_file_path)

# Loop through each training image for the current person
#for img_path in image_files_in_folder(os.path.join(image_dir)):
#    image = face_recognition.load_image_file(img_path)
    #print("Looking for faces in {}".format(image))
#    face_bounding_boxes = face_recognition.face_locations(image)

base_dir = os.path.dirname(__file__)

# # Create directory 'faces' if it does not exist
# if not os.path.exists('faces'):
# 	print("New directory created")
# 	os.makedirs('faces')

IMG_PATH = './faces/pic/'
face_count = 0
# Loop through all images and save images with marked faces
for file in os.listdir(base_dir + IMG_PATH):
    file_name, file_extension = os.path.splitext(file)
    if (file_extension in ['.png','.jpg','.jpeg']):
        print("Image path: {}".format(base_dir + IMG_PATH + file))

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(base_dir + IMG_PATH + file)

    height = image[0]
    width = image[1]

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)
    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    # if len(face_locations) != 1:
    #     os.remove(base_dir + './Face_rec_data/find_face/' + file)


    for face_location in face_locations:
        face_count += 1
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # top = top - 50
        # right = right + 50
        # bottom = bottom + 50
        # left = left - 50

        top = top if (top - 50) < 0 else (top - 50)
        right = right if (right + 50) < width.all() else (right + 50)
        #right = right if (right + 50) < 0 else (right + 50)
        bottom = bottom if (bottom + 50) < height.all() else (bottom + 50)
        #bottom = bottom if (bottom + 50) < 0 else (bottom + 50)
        left = left if (left - 50) < 0 else (left - 50)

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right] #[h,w]
        pil_image = Image.fromarray(face_image)
        data_time = datetime.now().strftime("%d_%m_%y|%H:%M:%S")
        #FaceFileName = os.path.join(IMG_PATH, data_time + "_" + str(face_count) + ".jpg")
        FaceFileName = os.path.join(IMG_PATH, "face" + "_" + str(face_count) + ".jpg")
        pil_image.save(FaceFileName)
        pil_image.show()
