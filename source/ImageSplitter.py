import os
import pickle
import zlib

import cv2
import face_recognition

class ImageSplitter:

    def saveDetectedFaces(self, image_path, save_path):

        try:

            os.stat(image_path)

        except Exception as e:

            print(f"Invalid image path :\n {e}")

            return -1

        try:

            if os.path.exists(save_path):

                os.path.isdir(save_path)

            else:

                os.makedirs(save_path, exist_ok=True)

        except Exception as e:

            print(f"Invalid save path :\n {e}")

            return -2

        rgb_image = face_recognition.load_image_file(image_path)
        detected_faces = face_recognition.face_locations(rgb_image)

        for face in detected_faces:

            extracted_face = rgb_image[face[0]:face[2], face[3]:face[1]]
            crc_hash = zlib.crc32(pickle.dumps(face) + pickle.dumps(extracted_face))

            cv2.imwrite(f"{save_path}/face_{crc_hash}.jpeg", extracted_face)
