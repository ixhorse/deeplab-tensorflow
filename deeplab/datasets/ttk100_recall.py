"""
recall of segmentation result of tt100k
"""

import os, sys
import numpy as np
import cv2 as cv
from glob import glob
import json
from tqdm import tqdm
import pdb

def get_box(annos, imgid):
    img = annos["imgs"][imgid]
    box_all = []
    for obj in img['objects']:
        box = obj['bbox']
        box = [int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])]
        box = [int(x * 0.3) for x in box]
        box_all.append(box)
    return box_all

def overlap(box1, box2):
    """
    Args:
        box1: prediction
        box2: ground truth
    """
    matric = np.array([box1, box2])
    u_xmin = np.max(matric[:,0])
    u_ymin = np.max(matric[:,1])
    u_xmax = np.min(matric[:,2])
    u_ymax = np.min(matric[:,3])
    u_w = u_xmax - u_xmin
    u_h = u_ymax - u_ymin
    if u_w <= 0 or u_h <= 0:
        return False
    u_area = u_w * u_h
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    if u_area / box1_area < 0.75:
        return False
    else:
        return True

def generate_box_from_mask(mask):
    box_all = []
    image, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[i])
        box_all.append([x, y, x+w, y+h])

    return box_all

def _test_generate_box_from_mask(mask, mask_box):
    ret, binary = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
    print(mask_box)
    for box in mask_box:
        cv.rectangle(binary, (box[0], box[1]), (box[2], box[3]), 100, 2)
    cv.imshow('a', binary)
    cv.waitKey(10000)

def enlarge_mask_box(mask_box, image_size):
    # enlarge 2x
    new_mask_box = []
    for box in mask_box:
        w = box[2] - box[0]
        h = box[3] - box[1]
        center_x = w / 2 + box[0]
        center_y = h / 2 + box[1]
        w = w * 1
        h = h * 1
        new_box = [center_x-w if center_x-w > 0 else 0,
                    center_y-h if center_y-h > 0 else 0,
                    center_x+w if center_x+w < image_size[0] else image_size[0]-1,
                    center_y+h if center_y+h < image_size[1] else image_size[1]-1]
        new_box = [int(x) for x in new_box]
        new_mask_box.append(new_box)
    return new_mask_box

def main():
    datadir = '/home/mcc/data/TT100K'
    label_path = datadir + '/TT100K_voc/SegmentationClass'
    annos_path = datadir + '/data/annotations.json'
    image_path = datadir + '/TT100K_voc/JPEGImages'
    mask_path = './tt100k/exp/vis/raw_segmentation_results'

    annos = json.loads(open(annos_path).read())

    total_object = 0
    detect_object = 0
    for raw_file in tqdm(glob(mask_path + '/*.png')):
        img_name = os.path.basename(raw_file)
        imgid = os.path.splitext(img_name)[0]
        label_file = os.path.join(label_path, img_name)
        image_file = os.path.join(image_path, imgid + '.jpg')
        
        mask_img = cv.imread(raw_file, cv.IMREAD_GRAYSCALE)
        # label_img = cv.imread(label_file, cv.IMREAD_GRAYSCALE)
        # original_img = cv.imread(image_file)
        mask_size = mask_img.shape[:2]

        label_box = get_box(annos, imgid)
        mask_box = generate_box_from_mask(mask_img)
        mask_box = enlarge_mask_box(mask_box, mask_size)
        # _test_generate_box_from_mask(mask_img, mask_box)
        # break

        count = 0
        for box1 in label_box:
            for box2 in mask_box:
                if overlap(box1, box2):
                    count += 1
                    break
        total_object += len(label_box)
        detect_object += count
    print('recall: %f' % (detect_object / total_object))

if __name__ == '__main__':
    main()