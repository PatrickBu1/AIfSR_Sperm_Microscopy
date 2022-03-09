import util.dataset
import cv2

def main():
    # data = util.dataset.SpermDataset(r"C:/Users/jnynt/Desktop/AifSR/data_claudia/augmented_v1_train/train", image_size=256)
    # image, label = data[0]
    image = cv2.imread('C:/Users/jnynt/Desktop/AifSR/test_data/random_images/dog.jpg')
    print(image.shape)
    cv2.imshow('_', image)
    cv2.waitKey()
    cv2.imshow('_', label)
    cv2.waitKey()

if __name__ == "__main__":
    main()