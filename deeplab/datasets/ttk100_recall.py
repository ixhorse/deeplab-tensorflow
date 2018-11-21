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


user_home = os.path.expanduser('~')
datadir = os.path.join(user_home, 'data/TT100K')
label_path = datadir + '/TT100K_voc/SegmentationClass'
annos_path = datadir + '/data/annotations.json'
image_path = datadir + '/TT100K_voc/JPEGImages'
mask_path = './tt100k/exp/vis/raw_segmentation_results'


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

def _boxvis(mask, mask_box):
    ret, binary = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
    print(mask_box)
    for box in mask_box:
        cv.rectangle(binary, (box[0], box[1]), (box[2], box[3]), 100, 2)
    cv.imshow('a', binary)
    key = cv.waitKey(0)
    sys.exit(0)

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

def vis_undetected_image(img_list):
    annos = json.loads(open(annos_path).read())

    for image in img_list:
        mask_file = os.path.join(mask_path, image+'.png')
        image_file = os.path.join(image_path, image+'.jpg')

        mask_img = cv.imread(mask_file, cv.IMREAD_GRAYSCALE)
        original_img = cv.imread(image_file)
        original_img[:,:,1] = np.clip(original_img[:,:,1] + mask_img*70, 0, 255)

        label_box = get_box(annos, image)
        for box in label_box:
            cv.rectangle(original_img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 1, 1)
        
        cv.imshow('1', original_img)
        key = cv.waitKey(1000*100)
        if key == 27:
            break

def main():
    annos = json.loads(open(annos_path).read())

    label_object = []
    detect_object = []
    mask_object = []
    undetected_img = []
    pixel_num = []
    for raw_file in tqdm(glob(mask_path + '/*.png')):
        img_name = os.path.basename(raw_file)
        imgid = os.path.splitext(img_name)[0]
        label_file = os.path.join(label_path, img_name)
        image_file = os.path.join(image_path, imgid + '.jpg')
        
        mask_img = cv.imread(raw_file, cv.IMREAD_GRAYSCALE)
        pixel_num.append(np.sum(mask_img))

        mask_size = mask_img.shape[:2]

        label_box = get_box(annos, imgid)
        mask_box = generate_box_from_mask(mask_img)
        mask_box = enlarge_mask_box(mask_box, mask_size)
        # _boxvis(mask_img, mask_box)
        # break

        count = 0
        for box1 in label_box:
            for box2 in mask_box:
                if overlap(box1, box2):
                    count += 1
                    break
        label_object.append(len(label_box))
        detect_object.append(count)
        mask_object.append(len(mask_box))
        if len(label_box) != count:
            undetected_img.append(imgid)

    print('recall: %f' % (np.sum(detect_object) / np.sum(label_object)))
    print('cost avg: %f, std: %f' % (np.mean(pixel_num), np.std(pixel_num)))
    print('detect box avg: %f, std %d' %(np.mean(mask_object), np.std(mask_object)))
    # print(undetected_img)

if __name__ == '__main__':
    img_list = ['1883', '75227', '38108', '29501', '48010', '15366', '91027', '31998', '25647', '29849', '77147', '82748', '78591', '25503', '75494', '36342', '13802', '1774', '66010', '83208', '51875', '27645', '74380', '12186', '37304', '25333', '50648', '40008', '28988', '44609', '18935', '9662', '36686', '68353', '75883', '41462', '2706', '26042', '35419', '14724', '6599', '29618', '51142', '80064', '75881', '77156', '65419', '37771', '66589', '4902', '1792', '16489', '81686', '89739', '70814', '58194', '42891', '88783', '12467', '40363', '76771', '36393', '5998', '90375', '97043', '90326', '2736', '22779', '36302', '79023', '79595', '77713', '72001', '84082', '84626', '94686', '38711', '69700', '66004', '94893', '94910', '94297', '7807', '70386', '94598', '71249', '70437', '20942', '41372', '40935', '92518', '4232', '39610', '97625', '97290', '96726', '58281', '80925', '62728', '38273', '41321', '67372', '68066', '5773', '25421', '76568', '56576', '58719', '45154', '94307', '31641', '47787', '12339', '73250', '46346', '62025', '59076', '2', '51435', '38199', '13931', '8783', '29297', '86217', '29714', '20600', '5051', '83167', '27986', '23320', '39338', '86066', '52335', '12883', '27663', '96461', '8099', '96069', '34782', '64497', '74315', '64195', '63555', '15332', '74518', '27009', '62713', '53500', '92707', '26234', '74668', '31643', '66117']
    # vis_undetected_image(img_list)
    main()