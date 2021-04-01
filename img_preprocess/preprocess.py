import numpy as np
import math
import cv2
import random
import os
from utils import load_config


# project setting
project_path = "../"
project = "line"

# load config
cfg = load_config(project_path)
cfg["train_data_dir"] = project_path + "dataset/" + project + '/train'
cfg["test_data_dir"] = project_path + "dataset/" + project + '/test'
cfg["prep_train_dir"] = project_path + 'preprocess/' + project + '/train'
cfg["prep_test_dir"] = project_path + 'preprocess/' + project + '/test'


def read_img(img_path, grayscale):
    """Read image"""
    if grayscale:
        im = cv2.imread(img_path, 0)
    else:
        im = cv2.imread(img_path)
    return im


def fill_squre(img, color):
    """fix image which not square"""
    height, width = img.shape[:2]
    max_val = max(height, width)
    if cfg["grayscale"]:
      size = (max_val, max_val)
      blank_image = np.zeros(size, np.uint8)
      blank_image[:, :] = color
    else:
      size = (max_val, max_val, 3)
      blank_image = np.zeros(size, np.uint8)
      blank_image[:, :] = (color, color, color)

    im = blank_image.copy()

    x_offset = y_offset = 0
    im[y_offset:y_offset + height, x_offset:x_offset + width] = img.copy()
    return im


def rotate_image(img, angle):
    """Rotate image"""
    h, w = img.shape[:2]
    angle %= 360
    M_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))
    return img_rotated


def random_rotate(img, angle_vari):
    """Random rotation degree"""
    angle = np.random.uniform(-angle_vari, angle_vari)
    return rotate_image(img, angle)


def is_image(filename):
    img_suffix = [".png", ".jpg", ".jpeg"]
    s = str.lower(filename)
    for suffix in img_suffix:
        if suffix == s[-4:]:
            return True
    return False


def check_files(dir_path, image_paths=None):
    if image_paths is None:
        image_paths = []
    files = [f for f in os.listdir(dir_path)]
    for f in files:
        if not os.path.isfile(os.path.join(dir_path, f)) and type(dir_path) == str:
            check_files(os.path.join(dir_path, f), image_paths)
        filepath = os.path.join(dir_path, f)
        if is_image(filepath):
            image_paths.append((filepath, 1))
    return image_paths


def generate_image_list(conf, is_train=True):
    """Generate all image data[path,] from target path"""
    if is_train:
        filenames = os.listdir(conf["train_data_dir"])
        num_imgs = len(filenames)
        num_ave_aug = int(math.floor(conf["augment_num"] / num_imgs))
        rem = conf["augment_num"] - num_ave_aug * num_imgs
        lucky_seq = [True] * rem + [False] * (num_imgs - rem)
        random.shuffle(lucky_seq)

        img_list = [(os.sep.join([conf["train_data_dir"], filename]), num_ave_aug + 1 if lucky else num_ave_aug)
                    for filename, lucky in zip(filenames, lucky_seq)]
    else:
        img_list = check_files(conf["test_data_dir"])
    return img_list


def augment_images(filelist, conf, is_train=True):
    """Generate augment images"""
    for filepath, n in filelist:
        img = read_img(filepath, conf["grayscale"])
        height, width = img.shape[:2]
        if (height, width) != (conf["img_resize"], conf["img_resize"]):
            if height != width and conf["fill_square"]:
                img = fill_squre(img, conf["fill_square"])
            img = cv2.resize(img, (conf["img_resize"], conf["img_resize"]))

        # Extract file names
        split_path = filepath.split(os.sep)
        filename = split_path[-1]
        sub_filename = split_path[-2]
        dot_pos = filename.rfind('.')
        imgname = filename[:dot_pos]
        ext = filename[dot_pos:]

        if is_train:
            print('Augmenting {} ...'.format(filename))
            for i in range(n):
                img_varied = img.copy()
                varied_imgname = '{}_{:0>3d}_'.format(imgname, i)

                if random.random() < conf["p_rotate"]:
                    img_varied_ = random_rotate(img_varied, conf["rotate_angle_vari"])
                    if img_varied_.shape[0] >= conf["img_resize"] and img_varied_.shape[1] >= conf["img_resize"]:
                        img_varied = img_varied_
                    varied_imgname += 'r'

                if random.random() < conf["p_horizonal_flip"]:
                    img_varied = cv2.flip(img_varied, 1)
                    varied_imgname += 'h'

                if random.random() < conf["p_vertical_flip"]:
                    img_varied = cv2.flip(img_varied, 0)
                    varied_imgname += 'v'

                output_filepath = os.path.join(conf["prep_train_dir"], '{}{}'.format(varied_imgname, ext))
                cv2.imwrite(output_filepath, img_varied)
        else:
            output_dir = cfg["prep_test_dir"]+"/"+sub_filename
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_filepath = os.path.join(output_dir, '{}{}'.format(imgname, ext))
            cv2.imwrite(output_filepath, img)


if __name__ == '__main__':
    # data augmentation
    if not os.path.exists(cfg["prep_train_dir"]):
        os.makedirs(cfg["prep_train_dir"])
        # train
        img_list = generate_image_list(cfg)
        augment_images(img_list, cfg)
        # test
        img_list = generate_image_list(cfg, is_train=False)
        augment_images(img_list, cfg, is_train=False)
    # train images
    print("Images(train-ok): ", len(os.listdir(cfg["prep_train_dir"])))
    # test images
    print("Images(test-ok): ", len(os.listdir(cfg["prep_test_dir"])))
