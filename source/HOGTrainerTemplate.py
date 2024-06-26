from os import listdir, path, makedirs
from time import perf_counter

from numpy import pad
from skimage.feature import hog
from face_recognition import load_image_file
from cv2 import resize, imwrite
from pickle import load

from .Logger import Logger

class HOGTrainerTemplate:

    def __init__(self, classifier = None, pixels_per_cell = 16, cells_per_block = 4, logger_name = "HOGTrainerTemplate"):

        self.hog_features = []
        self.images = []
        self.classes = []
        self.image_paths = []

        self.classifier = classifier
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

        self.max_dim1 = 0
        self.max_dim2 = 0
        self.max_dim1_changes = 0
        self.max_dim2_changes = 0

        self.logger = Logger(logger_name)
        self.logger.toggle_logger(True)
        self.logger = self.logger.logger

    def loadImagesHOG(self, images_path, visual_save_path = None):

        perf_start = perf_counter()

        for directory in listdir(images_path):

            self.logger.info(f"Iterating images in directory : {directory}")

            for img in listdir(path.join(images_path, directory)):

                self.logger.info(f"Loading image : {img}")

                im_path = path.join(images_path, directory, img)

                self.image_paths.append(im_path)

                image = load_image_file(im_path, mode = "L")
                imshape = image.shape

                self.logger.info(f"Loaded image dimensions : {imshape}")

                if (imshape[0] > self.max_dim1):

                    self.max_dim1 = imshape[0]
                    self.max_dim1_changes += 1

                    self.logger.info(f"New max_dim1 : {self.max_dim1}")

                if (imshape[1] > self.max_dim2):

                    self.max_dim2 = imshape[1]
                    self.max_dim2_changes += 1

                    self.logger.info(f"New max_dim2 : {self.max_dim2}")

                self.images.append(image)
                self.classes.append(directory)

        self.logger.info(f"Finished loading images in : {perf_counter() - perf_start}s")

        if ((self.max_dim1_changes > 1) and (self.max_dim2_changes > 1)):

            self.logger.info("Padding images with blank space for HoG training compatibility")

            perf_start = perf_counter()

            self.padImages(visual_save_path)
            self.logger.info(f"Time taken to pad all images and compute HoG : {perf_counter() - perf_start}s")

        else:

            self.logger.info("No images were padded due to same dimensions on every image")

    def padImages(self, visual_save_path):

        if (visual_save_path is None):

            visual_save_path = "hog_visualization"

            try:

                self.logger.info(f"Checking save path existence : {visual_save_path}")

                if path.exists(visual_save_path):

                    path.isdir(visual_save_path)

                else:

                    self.logger.info(f"Save path does not exist, creating : {visual_save_path}")
                    makedirs(visual_save_path, exist_ok=True)

            except Exception as e:

                self.logger.warning(f"Invalid save path :\n {e}")

                return -2

        for index, image in enumerate(self.images):

            padding = ((0, (self.max_dim1 - image.shape[0])), (0, (self.max_dim2 - image.shape[1])))

            self.logger.info(f"Padding image at index {index} of class {self.classes[index]} for {padding} dimensions. Image path {self.image_paths[index]}")

            self.images[index] = pad(image, pad_width=padding)

            self.logger.info(f"Computing HoG features for image at index {index} of class {self.classes[index]}. Image path {self.image_paths[index]}")

            perf_start = perf_counter()

            features, visualization = hog(self.images[index], orientations = 8,
                            pixels_per_cell = (self.pixels_per_cell, self.pixels_per_cell),
                            cells_per_block = (self.cells_per_block, self.cells_per_block),
                            block_norm = "L2", visualize = True)
            save_path = path.join(visual_save_path, self.image_paths[index].split('\\')[-1])

            self.logger.info(f"HoG features for image at index {index} of class {self.classes[index]} computed in {perf_counter() - perf_start}s. Image path {self.image_paths[index]}")
            self.hog_features.append(features)
            self.logger.info(f"Saving HoG visualization for image at index {index} of class {self.classes[index]}. Image path : {self.image_paths[index]}. Save path : {save_path}")
            imwrite(save_path, visualization)

    def predictFace(self, image_path,  model_path = None, classifier = None):

        if not path.isfile(image_path):

            exception_str = f"The given image path is not an image file : {image_path}"

            self.logger.error(exception_str)
            raise Exception(exception_str)

        if model_path is None and classifier is None and self.classifier is None:

            exception_str = "Provide model_path / trained KNeighborsClassifier object / train the built in classifier before running predict"

            self.logger.error(exception_str)
            raise Exception(exception_str)

        if model_path is None and classifier is None:

            classifier = self.classifier

            self.logger.info("Setting classifier model to the current object internal classifier")

        if model_path is None and self.classifier is None:

            classifier = classifier

            self.logger.info("Setting classifier model to the classifier object provided in the arguments")

        if classifier is None and self.classifier is None:

            with open(model_path, "rb") as model:

                classifier = load(model)

            self.logger.info("Setting classifier to the classifier object provided in the classifier model path")

        self.logger.info(f"Loading image from {image_path}")

        image = load_image_file(image_path, mode = "L")

        if ((image.shape[0] >= classifier.max_dim1) or (image.shape[1] >= classifier.max_dim2)):

            self.logger.info(f"Resizing image loaded from {image_path} to dimensions {(classifier.max_dim2, classifier.max_dim1)} to fit HoG features")

            image = resize(image, dsize = (classifier.max_dim2, classifier.max_dim1))

        self.logger.info(f"Padding image at {image_path} with {((0, (classifier.max_dim1 - image.shape[0])), (0, (classifier.max_dim2 - image.shape[1])))}")

        image = pad(image, pad_width=((0, (classifier.max_dim1 - image.shape[0])), (0, (classifier.max_dim2 - image.shape[1]))))

        self.logger.info(f"Computing HoG features for image loaded from {image_path}")

        start_time = perf_counter()
        features = hog(image, orientations = 8,
                            pixels_per_cell = (self.pixels_per_cell, self.pixels_per_cell),
                            cells_per_block = (self.cells_per_block, self.cells_per_block),
                            block_norm = "L2")

        self.logger.info(f"Computed HoG features for image at {image_path} in {perf_counter() - start_time}s")
        self.logger.info(f"Predicting class of image loaded from {image_path}")

        start_time = perf_counter()

        prediction = classifier.predict([features])[0]

        self.logger.info(f"Predicted class for image loaded from {image_path} as {prediction} in {perf_counter() - start_time}s")

        return prediction
