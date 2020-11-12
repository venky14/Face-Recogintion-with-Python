"""
# This is a demo of running face recognition on a video file and saving the results to a new video file.
"""

import cv2
import face_recognition


# Open the input movie file
#input_movie = cv2.VideoCapture("./vid_sample_test/Narendra_Modi_clip.mp4")
input_movie = cv2.VideoCapture("./vid_sample_test/Modi_Akshay.mp4")

namo_sample_pic = "./images_dataset/knn_train_data/Narendra_Modi/narendramodi-pti.jpg"
ak_sample_pic = "./images_dataset/knn_train_data/Akshay_Kumar/ak22.jpeg"

namo_label = "Narendra Modi"
ak_label = "Akshay Kumar"

length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#output_movie = cv2.VideoWriter('namo_output2.avi', fourcc, 13, (320, 240))
output_movie = cv2.VideoWriter('./vid_sample_test/vid_output2.avi', fourcc, 50, (1280, 720)) #w,h


def facerec_video_output():
    # Load some sample pictures and learn how to recognize them.
    namo_image = face_recognition.load_image_file(namo_sample_pic)
    namo_face_encoding = face_recognition.face_encodings(namo_image)[0]

    ak_image = face_recognition.load_image_file(ak_sample_pic)
    ak_face_encoding = face_recognition.face_encodings(ak_image)[0]

    known_faces = [
        namo_face_encoding,
        ak_face_encoding
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0

    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        if frame_number == 1:

            # Quit when the input video file ends
            if not ret:
                break

            if ret == True:

                cv2.imshow('captured_frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which dlib, face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

                # If you had more than 2 faces, you could make this logic a lot prettier
                # but I kept it simple for the demo
                name = None
                if match[0]:
                    name = "Narendra Modi"
                elif match[1]:
                    name = "Akshay Kumar"

                face_names.append(name)

            # Label the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if not name:
                    continue

                # Draw a box around the face
                #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 200), 2)

                # # Draw a label with a name below the face
                # cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 10), (right, bottom), (0, 0, 200), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.3, (255, 255, 255), 1)


            # Write the resulting image to the output video file
            print("Writing frame {} / {}".format(frame_number, length))
            frame_number = 0
            output_movie.write(frame)

    # All done!
    input_movie.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

	facerec_video_output()
