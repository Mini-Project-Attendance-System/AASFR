import os

from source import ImageSplitter

if __name__ == "__main__":

    splitter = ImageSplitter()
    root = "./preprocessed_testing_images"
    save = "./testing_images"

    for image in os.listdir(root):

        splitter.findFaces(os.path.join(root, image), save)
