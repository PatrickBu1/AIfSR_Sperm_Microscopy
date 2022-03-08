import albumentations as A
import argparse
import cv2
import os
import numpy as np

'''
0. flip diagonally
1. rotate left
2. rotate right
3. crop to 400x400
4. grid distort
5. gaussian blur
6. gaussian noise with shiftscalerotate

'''


def main(i_src, l_src, dst, i_suffix, l_suffix):
    if not os.path.exists(dst+'/images'):
        os.mkdir(dst + "/images")
    if not os.path.exists(dst+'/labels'):
        os.mkdir(dst + "/labels")

    flip = A.Compose([A.Flip(p=1)])
    left_rotate = A.Compose([A.Rotate(limit=(-90, -90), p=1.0)])
    right_rotate = A.Compose([A.Rotate(limit=(90, 90), p=1.0)])
    crop = A.Compose([A.CenterCrop(height = 256, width = 256, always_apply=False, p=1.0)])
    grid_distort = A.Compose([A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1,
                            border_mode=4, value=None, mask_value=None, always_apply=False, p=1.0)])
    gaussian_blur = A.Compose([A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0, always_apply=False, p=1.0)])
    scr_gauss = A.Compose([A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45,
                            interpolation=1, border_mode=4, value=None, mask_value=None,
                            shift_limit_x=None, shift_limit_y=None, always_apply=False, p=1.0),
                           A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=1.0)
                           ])

    image_list = []
    label_list = []

    for root, dirs, files in os.walk(i_src):
        for file in files:
            image_list.append(int(file[:-len(i_suffix)]))

    for root, dirs, files in os.walk(l_src):
        for file in files:
            label_list.append(int(file[:-len(l_suffix)]))

    image_list.sort()
    label_list.sort()
    output = []

    for i in range(len(image_list)):
        img = cv2.imread(i_src + str(image_list[i]) + i_suffix)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(l_src + str(label_list[i]) + l_suffix)
        output.append(flip(image=img, mask=lbl))
        output.append(left_rotate(image=img, mask=lbl))
        output.append(right_rotate(image=img, mask=lbl))
        output.append(crop(image=img, mask=lbl))
        output.append(grid_distort(image=img, mask=lbl))
        output.append(gaussian_blur(image=img, mask=lbl))
        output.append(scr_gauss(image=img, mask=lbl))
        for j in range(len(output)):
            out_img = output[j]['image']
            out_mask = output[j]['mask']
            if out_img.shape[0] != 512:
                out_img = cv2.resize(out_img, (512, 512))
                out_mask = cv2.resize(out_mask, (512, 512))
            cv2.imwrite(dst + '/images/' + str(j) + i_suffix, out_img)
            cv2.imwrite(dst + '/labels/' + str(j) + l_suffix, out_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input and output directory')
    parser.add_argument('image_source', type=str, help='source folder')
    parser.add_argument('label_source', type=str, help='source folder')
    parser.add_argument('destination', type=str, help='destination folder')
    parser.add_argument('i_suffix', type=str, help='image file suffix')
    parser.add_argument('l_suffix', type=str, help='label file suffix')
    args = parser.parse_args()
    main(args.image_source, args.label_source, args.destination, args.i_suffix, args.l_suffix)
    '''
    image_source = "C:/Users/jnynt/Desktop/AifSR/data_claudia/Dataset with generated masks/images/"
    label_source = "C:/Users/jnynt/Desktop/AifSR/data_claudia/Dataset with generated masks/labels/"
    destination = "C:/Users/jnynt/Desktop/AifSR/data_claudia/augmented_v1"
    main(image_source, label_source, destination, ".tif", ".png")
    '''
