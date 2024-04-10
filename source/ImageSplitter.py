from os import path, makedirs, stat
from pickle import dumps
from time import perf_counter
from zlib import crc32

from cv2 import imwrite
from face_recognition import load_image_file, face_locations

from .Logger import Logger

class ImageSplitter:

    def __init__(self):

        self.logger = Logger("ImageSplitter")
        self.logger.toggle_logger(True)
        self.logger = self.logger.logger

    def findFaces(self, image_path, save_path = None):

        extracted_faces = []
        perf_start = 0
        perf_end = 0

        try:

            self.logger.info(f"Checking validity of path : {image_path}")
            stat(image_path)

        except Exception as e:

            self.logger.warning(f"Invalid image path :\n {e}")

            return -1

        if (save_path is not None):

            try:

                self.logger.info(f"Checking save path existence : {save_path}")

                if path.exists(save_path):

                    path.isdir(save_path)

                else:

                    self.logger.info(f"Save path does not exist, creating : {save_path}")
                    makedirs(save_path, exist_ok=True)

            except Exception as e:

                self.logger.warning(f"Invalid save path :\n {e}")

                return -2

        self.logger.info(f"Processing image : {image_path}")

        perf_start = perf_counter()
        rgb_image = load_image_file(image_path)
        detected_faces = face_locations(rgb_image)
        perf_end = perf_counter()

        self.logger.info(f"Found {detected_faces.__len__()} faces in {image_path}")
        self.logger.info(f"Time taken to detect faces from {image_path} : {perf_end - perf_start}s")

        for face in detected_faces:

            extracted_face = rgb_image[face[0]:face[2], face[3]:face[1]]

            self.logger.info(f"Processing found face at location : {face} in {image_path}")

            if (save_path is None):

                extracted_faces.append(extracted_face)

            if (save_path is not None):

                crc_hash = crc32(dumps(face) + dumps(extracted_face))

                self.logger.info(f"Saving {image_path} to : {save_path} as face_{crc_hash}")
                imwrite(f"{save_path}/face_{crc_hash}.jpeg", extracted_face)

        return extracted_faces
