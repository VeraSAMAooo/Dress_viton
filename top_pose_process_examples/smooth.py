import cv2
from PIL import Image, ImageFilter
import numpy as np
from copy import deepcopy
import torch
from torch.nn import functional as F
import torch.nn as nn

def post_process_viton_top_image(result_raw_image,origin_c_image,b_parse_image,agnostic_mask,
                                 ori_result_parse_image,labels,input_h=1024):
    #### ori_parse_image is the atr parse of the viton result
    #### map 18 atr labels to 13 parse labels
    ori_result_parse_image = np.asarray(ori_result_parse_image)
    parse_image = np.zeros((ori_result_parse_image.shape[0], ori_result_parse_image.shape[1]))
    for i in range(len(labels)):
        for label in labels[i][1]:
            parse_image[np.where(ori_result_parse_image == label)] = i

    ##### get face mask from origin image atr parse(already processed face mask with lip parse result )
    ##b_parse: parse of the origin image(cut the neck already), 3 and 11 from 18 channels
    face_mask = ((b_parse_image == 3).astype(np.float32) + (b_parse_image == 11).astype(np.float32))

    #### get upper, left_arm,right_arm, left_leg,right_leg, neck(atr parse of face mask - lip parse of face mask) mask from viton result parse
    target_mask =((parse_image == 2).astype(np.float32) - (face_mask == 1).astype(np.float32) )
    # target_mask = Image.fromarray(np.uint8(target_mask * 255))
    # target_mask = target_mask.filter(ImageFilter.BoxBlur(5))
    # target_mask = target_mask.save('/home/vera/myCode/top_pose_process_examples/smooth_test/output2/yy.png')

    # target_mask = target_mask * 255
    # kernel = np.ones((5, 5), np.float32) / 25
    # target_mask_filter = cv2.filter2D(target_mask, -1, kernel)
    #
    im = result_raw_image.copy()
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)

    im = Image.fromarray(img)
    #
    # result_raw_image = cv2.cvtColor(result_raw_image, cv2.COLOR_BGR2RGB)
    # result_raw_image = Image.fromarray(result_raw_image)
    # result_raw_image.paste((im, None, target_mask).filter(ImageFilter.BoxBlur(50)))
    #
    white = Image.new("RGB", (768 , 1024), (255, 255, 255))
    #
    white2 = white.copy()
    white.paste(im, None, Image.fromarray(np.uint8(target_mask * 255), 'L'))
    white = white.filter(ImageFilter.GaussianBlur(2))

    # white_png.save('/home/vera/myCode/top_pose_process_examples/smooth_test/output2/pp.png')

    img1 = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)

    img2 = cv2.cvtColor(np.asarray(white),cv2.COLOR_RGB2BGR)

    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 120, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)  #

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    cv2.imshow('res', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows



    #
    # img2_fg = cv2.bitwise_and(white, white, mask=mask_png)
    # dst = cv2.add(white2, img2_fg)
    #
    # cv2.imwrite('/home/vera/myCode/top_pose_process_examples/smooth_test/output2/mask_dst.png', dst)
    # white = white.filter(ImageFilter.BoxBlur(2))
    #get the mask from white and paste it to raw image


    return white

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
    pose_name = '10'
    result_list = os.listdir('/home/vera/myCode/top_pose_process_examples/smooth_test/viton_result')
    result_raw_image_dir = '/home/vera/myCode/top_pose_process_examples/smooth_test/viton_result/'
    result_parse_image_dir = '/home/vera/myCode/top_pose_process_examples/smooth_test/viton_result_parse_atr/'
    origin_c_image_path = '/home/vera/myCode/top_pose_process_examples/smooth_test/image/{}.jpg'.format(pose_name)
    b_parse_image_path = '/home/vera/myCode/top_pose_process_examples/smooth_test/image-parse-v3-atr/{}.png'.format(pose_name)
    agnostic_mask_path = '/home/vera/myCode/top_pose_process_examples/smooth_test/agnostic_mask/{}.png'.format(pose_name)
    input_h = 1024

    origin_c_image = cv2.imread(origin_c_image_path)
    b_parse_image = Image.open(b_parse_image_path)
    b_parse_image = np.asarray(b_parse_image)
    agnostic_mask = Image.open(agnostic_mask_path)
    agnostic_mask = np.asarray(agnostic_mask)

    des_dir = '/home/vera/myCode/top_pose_process_examples/smooth_test/output2'


    if not os.path.exists(des_dir):
        os.makedirs(des_dir)


    for i in range(len(result_list)):

        result_raw_image = cv2.imread(result_raw_image_dir + result_list[i] )

        ori_result_parse_image = Image.open(result_parse_image_dir + result_list[i])
        ori_result_parse_image = np.asarray(ori_result_parse_image)
        result = post_process_viton_top_image(result_raw_image,origin_c_image,b_parse_image,agnostic_mask,ori_result_parse_image,top_labels,input_h=1024)
        # cv2.imwrite(os.path.join(des_dir, result_list[i]), result)
        #cv2 to pil
        # img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        # img = img.astype(np.uint8)
        # result = Image.fromarray(img)
        #
        # result_raw_image = cv2.cvtColor(result_raw_image, cv2.COLOR_BGR2RGB)
        # result_raw_image = Image.fromarray(result_raw_image)
        #
        # #parse arms and legs on the result
        # parse_array = np.array(ori_result_parse_image)
        # parse_dress = (parse_array == 7).astype(np.float32)
        # result.paste(result_raw_image, None, Image.fromarray(np.uint8(parse_dress * 255), 'L'))
        # result = result.save(os.path.join(des_dir, result_list[i].replace('.png','4.png')))


