import cv2
from PIL import Image
import numpy as np
from copy import deepcopy
import torch
from torch.nn import functional as F
import torch.nn as nn



if __name__ == '__main__':
    import glob
    import os
    import json
    top_labels = {
        0: ['background', [0]],
        1: ['hair', [1, 2]],
        2: ['face', [3, 11]],
        3: ['dress', [4, 5, 6, 7, 8]],
        4: ['left_arm', [14]],
        5: ['right_arm', [15]],
        6: ['left_leg', [12]],
        7: ['right_leg', [13]],
        8: ['left_shoe', [9]],
        9: ['right_shoe', [10]],
        10: ['bag', [16]],
        11: ['noise', [17]]
    }

    ori_img = Image.open('/home/vera/myCode/top_pose_process_examples/set_0309/image/green.jpg')
    # agnostic_img = cv2.imread('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0227/result/model3/agnostic/3.png', cv2.IMREAD_UNCHANGED)
    res_list = os.listdir('/home/vera/myCode/top_pose_process_examples/set_0309/viton_result')
    res_img_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/viton_result/'
    # res_img_dir2 = '/home/vera/myCode/top_pose_process_examples/set_0309/viton_result/'
    res_parse_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/viton_result_parse_atr/'
    # res_parse_dir2 = '/home/vera/myCode/top_pose_process_examples/set_0309/viton_result_parse_atr/'
    white = Image.new("RGB", (768, 1024), (255, 255, 255))
    #
    # mask = agnostic_img[:, :, 3]
    # cv2.imshow("1.jpg", mask)
    des_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/all_res/green/green-res-body/green-midi-512'
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    for i in range(len(res_list)):
        print(res_list[i])
        img = ori_img.copy()
        # white = Image.new("RGB", (768, 1024), (255, 255, 255))

        res_img = Image.open(res_img_dir + res_list[i])
        # res_img2 = Image.open(res_img_dir2 + res_list[i])
        # res_img[np.where(agnostic_img) < 128] = 255
        res_parse = Image.open(res_parse_dir + res_list[i].replace('out.jpg', 'out.png'))
        # res_parse2 = Image.open(res_parse_dir2 + res_list[i].replace('out.jpg', 'out.png'))

        # res_parse[np.where(agnostic_img < 128)] = 255
        parse_array = np.array(res_parse)
        # parse_array2 = np.array(res_parse2)

        # parse_array[np.where(agnostic_img) < 128] = 255

        parse_dress = ((parse_array == 7).astype(np.float32)+
                       (parse_array == 4).astype(np.float32)+
                       (parse_array == 5).astype(np.float32)+
                       (parse_array == 6).astype(np.float32)+
                       (parse_array == 8).astype(np.float32))
        parse_dress2 = ((parse_array == 7).astype(np.float32)+
                       (parse_array == 4).astype(np.float32)+
                       (parse_array == 5).astype(np.float32)+
                       (parse_array == 6).astype(np.float32)+
                       (parse_array == 8).astype(np.float32))

        img.paste(res_img, None, Image.fromarray(np.uint8(parse_dress * 255), 'L'))
        # img.paste(res_img, None, Image.fromarray(np.uint8(parse_dress2 * 255), 'L'))

        # white.paste(res_img, None, Image.fromarray(np.uint8(parse_dress * 255), 'L'))

        img = img.save(os.path.join(des_dir, res_list[i]))
        # white = white.save(os.path.join(des_dir, res_list[i]))
