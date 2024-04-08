from os import listdir

from source import KNNTrainer

if __name__ == "__main__":

    predictor = KNNTrainer()

    for image in listdir("testing_images"):

        try:

            prediction = predictor.predictFace(f"testing_images/{image}", "trained_models/vakibc.clf")
            print(prediction)

        except Exception as e:

            print(f"{e}\nOccured for image : {image}")

            continue
