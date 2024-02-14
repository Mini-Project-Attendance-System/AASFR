import os
import pickle
import zlib

import cv2
import dlib

# * ImageSplitter uses 8-bit color depth for any image fed into it
class ImageSplitter:
    """
    ImageSplitter is a utility class that helps extract individual faces
    from an RGB image containing faces of multiple people present with
    their faces turned to the camera

    * Implements functions provided by dlib library to detect faces and crop
    * out the detected faces
    """

    def __init__(self):

        self.face_detector = dlib.get_frontal_face_detector()

    def getFaceBoundingBoxes(self, rgb_image):
        """
        getFaceBoundingBoxes takes in an RGB Image and spits out multiple
        rectangular bounding boxes containing just the face of each person
        detected in the image

        Params:
            rgb_image:
                numpy.ndarray : A four dimensional array containing rgb image data
                obtained from dlib.load_rgb_image function


        Returns:
            ! Returns empty collection on error

            dlib.rectangles : A collection of dlib.rectangle objects which
            enclose bounds of every face detected in the image

        """

        assert rgb_image is not None

        try:

            return self.face_detector(rgb_image, 1)

        except Exception as e:

            print(f"Exception warning when detecting faces :\n {e}")

            return []

    def saveDetectedFaces(self, image_path, save_path):
        """
        getDetectedFaces takes in a path to an RGB image and saves the detected
        faces into the filesystem

        Params:
            image_path:
                str : Path to an RGB image in the filesystem

            save_path:
                str: Path to a folder in the filesystem to save detected faces

        Returns:
            ! Returns a value of -1 when image path is invalid
            ! Returns a value of -2 when save path is invalid
        """

        assert image_path is not None
        assert save_path is not None

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

        rgb_image = dlib.load_rgb_image(image_path)
        detected_faces = self.getFaceBoundingBoxes(rgb_image)

        for face in detected_faces:

            extracted_face = dlib.sub_image(rgb_image, face)
            crc_hash = zlib.crc32(pickle.dumps(face) + pickle.dumps(extracted_face))

            cv2.imwrite(f"{save_path}/face_{crc_hash}.jpeg", extracted_face)
