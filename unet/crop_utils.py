import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def join_mask(step, image, crop_folder, save_folder, ch):
    out = os.listdir(crop_folder)
    path = "data/stage1_test/{}/images/{}.png".format(image, image)
    crops = get_sorted_names(image, out, ch)
    mask = cv2.imread(path)
    height, width, _ = mask.shape
    matrices = []
    zeros = np.zeros((height, width))
    cnt = 0
    cnt_h = 0
    while (height > 0):
        if height > step:
            step_h = step
            b = (cnt_h + 1) * step_h
            a = cnt_h * step_h
        else:
            step_h = step
            b = mask.shape[0]
            a = mask.shape[0] - step_h
        width = mask.shape[1]
        cnt_w = 0
        while width > 0:
            if width > step:
                step_w = step
                read_path = os.path.join(crop_folder, crops[cnt])
                zeros[a:b, cnt_w * step_w:(cnt_w + 1) * step_w] = cv2.imread(read_path, 0)

            else:
                step_w = step
                read_path = os.path.join(crop_folder, crops[cnt])
                zeros[a:b, (mask.shape[1] - step_w):mask.shape[1]] = cv2.imread(read_path, 0)
            cnt_w += 1
            cnt += 1
            matrices.append(zeros)

            width = width - step_w
        cnt_h += 1
        height = height - step_h
    cv2.imwrite('{}/{}_{}.png'.format(save_folder, image, ch), zeros)

def get_sorted_names(name, file_list, ch):
    outs = [i.split('.', 1)[0] for i in out]
    np_out = np.array([i.split('_', 3) for i in outs])
    zero_st = [(np_out[:, 0] == name) & (np_out[:, -1] == ch)]
    stuff = np.array(file_list)[zero_st]
    return sorted(stuff)

def split_mask(step, image, save_folder):
    # path = "data/stage1_train_/{}/masksmask.png".format(image)
    path = "data/stage1_test/{}/images/{}.png".format(image, image)
    mask = cv2.imread(path)
    print(mask.shape)
    height, width, _ = mask.shape
    matrices = []
    cnt_h = 0
    while height > 0:
        if height > step:
            step_h = step
            b = (cnt_h+1)*step_h
            a = cnt_h*step_h
        else:
            step_h = step
            b = mask.shape[0]
            a = mask.shape[0]-step_h

        width = mask.shape[1]
        cnt_w = 0
        while width > 0:
            zeros = np.zeros((step, step))
            if width > step:
                step_w = step
                zeros = mask[a:b, cnt_w*step_w:(cnt_w+1)*step_w]
            else:
                step_w = step
                zeros = mask[a:b, (mask.shape[1] - step_w):mask.shape[1]]
            cnt_w += 1
            matrices.append(zeros)
            cv2.imwrite('{}/{}_{}_{}.png'.format(save_folder, image, cnt_h, cnt_w), zeros)
            width = width - step_w
        cnt_h +=1
        height = height - step_h
    print(len(matrices))


if __name__ == '__main__':
    # split
    imgs = os.listdir('data/stage1_test/')
    [split_mask(128, img, 'data/cropped_test/') for img in imgs]

    # join
    # [join_mask(128, img, 'output/mask/', 'output/joined_mask/', '0') for img in imgs]