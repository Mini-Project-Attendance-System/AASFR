from os import listdir, path
from random import shuffle

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from source import KNNTrainer, Logger

if __name__ == "__main__":

    predictor = KNNTrainer()

    logger = Logger("KNNPrediction")
    logger.toggle_logger(True)
    logger = logger.logger

    root = "testing_images"

    predictions = []
    labels = []
    paths = []

    def logMetrics():

        logger.info(f"Accuracy score : {accuracy_score(labels, predictions)}")
        logger.info(f"Classification report :\n{classification_report(labels, predictions)}")

        matrix = confusion_matrix(labels, predictions)

        logger.info(f"Confusion matrix :\n{matrix}")

    logger.info("Run 1 with sequential data feed without random passes to predictor")

    for directory in listdir(root):

        if (directory == "_group_pics"):

            continue

        for image in listdir(path.join(root, directory)):

            paths.append(path.join(root, directory, image))
            labels.append(directory)

            prediction = predictor.predictFace(path.join(root, directory, image), "trained_models/vakibc.clf")

            predictions.append(prediction)

    logMetrics()
    logger.info("Run 2 with data scrambled randomly")

    predictions = []
    zipped_data = list(zip(paths, labels))

    shuffle(zipped_data)

    paths, labels = zip(*zipped_data)

    for path in paths:

        prediction = predictor.predictFace(path, "trained_models/vakibc.clf")

        predictions.append(prediction)

    logMetrics()
