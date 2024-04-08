from os import listdir, path

from numpy import pad
from skimage.feature import hog
from face_recognition import load_image_file
from cv2 import resize
from pickle import load

class HOGTrainerTemplate:

    def __init__(self, classifier = None, pixels_per_cell = 16, cells_per_block = 4):

        self.hog_features = []
        self.images = []
        self.classes = []
        self.classifier = classifier
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.max_dim1 = 0
        self.max_dim2 = 0

    def loadImagesHOG(self, images_path):

        for directory in listdir(images_path):

            print(f"Iterating images in directory : {directory}")

            for img in listdir(path.join(images_path, directory)):

                print(f"Loading image : {img}")

                image = load_image_file(path.join(images_path, directory, img), mode = "L")
                imshape = image.shape

                if (imshape[0] > self.max_dim1):

                    self.max_dim1 = imshape[0]

                if (imshape[1] > self.max_dim2):

                    self.max_dim2 = imshape[1]
                self.images.append(image)
                self.classes.append(directory)

        self.padImages()

    def padImages(self):

        for index, image in enumerate(self.images):

            self.images[index] = pad(image, pad_width=((0, (self.max_dim1 - image.shape[0])), (0, (self.max_dim2 - image.shape[1]))))

            features = hog(self.images[index], orientations = 8,
                            pixels_per_cell = (self.pixels_per_cell, self.pixels_per_cell),
                            cells_per_block = (self.cells_per_block, self.cells_per_block),
                            block_norm = "L2")

            self.hog_features.append(features)

    def predictFace(self, image_path,  model_path = None, classifier = None):

        if not path.isfile(image_path):

            raise Exception("The given image path is not an image file : ", image_path)

        if model_path is None and classifier is None and self.classifier is None:

            raise Exception("Provide model_path / trained KNeighborsClassifier object /\
                train the built in classifier before running predict")

        if model_path is None and classifier is None:

            classifier = self.classifier

        if model_path is None and self.classifier is None:

            classifier = classifier

        if classifier is None and self.classifier is None:

            with open(model_path, "rb") as model:

                classifier = load(model)

        image = load_image_file(image_path, mode = "L")

        if ((image.shape[0] >= classifier.max_dim1) or (image.shape[1] >= classifier.max_dim2)):

            image = resize(image, dsize = (classifier.max_dim2, classifier.max_dim1))

        image = pad(image, pad_width=((0, (classifier.max_dim1 - image.shape[0])), (0, (classifier.max_dim2 - image.shape[1]))))
        features = hog(image, orientations = 8,
                            pixels_per_cell = (self.pixels_per_cell, self.pixels_per_cell),
                            cells_per_block = (self.cells_per_block, self.cells_per_block),
                            block_norm = "L2")

        prediction = classifier.predict([features])[0]

        return f"Prediction for {image_path} : {prediction}"
