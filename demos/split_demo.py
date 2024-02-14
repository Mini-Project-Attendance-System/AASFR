import os

from source import ImageSplitter

if __name__ == "__main__":

    splitter = ImageSplitter()

    for image in os.listdir("./test_images"):

        splitter.saveDetectedFaces(f"./test_images/{image}", f"./demo_out/{image}")
