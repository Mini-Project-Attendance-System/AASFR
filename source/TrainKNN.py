from os import listdir, path
from math import sqrt
from pickle import dump, load

from cv2 import imshow, namedWindow, waitKey
from numpy import array
from sklearn import neighbors
from face_recognition import face_encodings, face_locations, load_image_file

class KNNTrainer:

    def __init__(self):

        self.images = []
        self.classes = []
        self.classifier_knn = None

    def loadImageEncodings(self, images_path):

        images = []

        for directory in listdir(images_path):

            print(f"Iterating images in directory : {directory}")

            for img in listdir(path.join(images_path, directory)):

                print(f"Loading image : {img}")

                image = load_image_file(path.join(images_path, directory, img))
                face_boxes = face_locations(image)

                print(f"Encoding Facial Features of image : {img}")

                encoded_face = face_encodings(image, known_face_locations=face_boxes)[0]

                self.images.append(encoded_face)
                self.classes.append(directory)

    def trainKNN(self, n_neighbors = None, save_path = None):

        if n_neighbors is None:

            n_neighbors = int(round(sqrt(len(self.images))))

        self.classifier_knn = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, algorithm="ball_tree", weights="distance")

        self.classifier_knn.fit(array(self.images), array(self.classes))

        if save_path is not None:

            with open(save_path, "wb+") as knn_model:

                dump(self.classifier_knn, knn_model)

    def predictFace(self, image_path,  model_path = None, classifier_knn = None, threshold = 0.6):

        if not path.isfile(image_path):

            raise Exception("The given image path is not an image file : ", image_path)

        if model_path is None and classifier_knn is None and self.classifier_knn is None:

            raise Exception("Provide model_path / trained KNeighborsClassifier object /\
                train the built in classifier before running predict")

        if model_path is None and classifier_knn is None:

            classifier = self.classifier_knn

        if model_path is None and self.classifier_knn is None:

            classifier = classifier_knn

        if classifier_knn is None and self.classifier_knn is None:

            with open(model_path, "rb") as model:

                classifier = load(model)

        img = load_image_file(image_path)
        bounds = face_locations(img)
        encoding = face_encodings(img, known_face_locations=bounds)

        closest_dists = classifier.kneighbors(encoding, n_neighbors=1)
        matches = [closest_dists[0][i][0] <= threshold for i in range(len(bounds))]

        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(classifier.predict(encoding), bounds, matches)]
