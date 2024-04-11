from os import listdir, path

from sklearn.metrics import classification_report, accuracy_score

from source import KNNTrainer, Logger

if __name__ == "__main__":

    predictor = KNNTrainer()

    logger = Logger("KNNPrediction")
    logger.toggle_logger(True)
    logger = logger.logger

    root = "testing_images"

    predictions = []
    labels = []

    for directory in listdir(root):

        for image in listdir(path.join(root, directory)):

            labels.append(directory)

            prediction = predictor.predictFace(path.join(root, directory, image), "trained_models/vakibc.clf")

            predictions.append(prediction)

    logger.info(f"Accuracy score : {accuracy_score(labels, predictions)}")
    logger.info(f"Classification report :\n{classification_report(labels, predictions)}")
