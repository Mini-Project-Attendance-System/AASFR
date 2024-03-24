from source import KNNTrainer

if __name__ == "__main__":

    trainer = KNNTrainer()

    trainer.loadImageEncodings("training_images/")
    trainer.trainKNN(n_neighbors=2, save_path="trained_models/vakibc.clf")
