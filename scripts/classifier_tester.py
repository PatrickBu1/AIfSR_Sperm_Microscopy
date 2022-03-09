import torch
import cv2
from torchvision import models as torchmodel
from torchvision import transforms
import models.v_unet

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def infer_once(model, img_dir):
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (512, 512))
    img = transform(img)
    model.eval()
    device = torch.device('cpu')
    img = img.to(device)
    output = model(img[None, ...])
    result = torch.argmax(output).item()
    print(result)


def main():
    img_dir = 'C:/Users/jnynt/Desktop/AifSR/test_data/random_images/dog.jpg'
    model = models.v_unet.VUnet(pretrained_path=r'C:/Users/jnynt/Desktop/AifSR/model_pth/vgg16_classifier_parsed.pth')
    # model = torchmodel.vgg16(pretrained=True, progress=True)
    infer_once(model, img_dir)


if __name__ == '__main__':
    main()