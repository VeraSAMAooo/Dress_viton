import cv2
from PIL import Image, ImageFilter
import numpy as np

def post_process_viton_top_image(result_raw_image,origin_c_image,b_parse_image,agnostic_mask,
                                 ori_result_parse_image,labels,input_h=1024):
    #### ori_parse_image is the atr parse of the viton result
    #### map 18 atr labels to 13 parse labels
    ori_result_parse_image = np.asarray(ori_result_parse_image)
    parse_image = np.zeros((ori_result_parse_image.shape[0], ori_result_parse_image.shape[1]))
    for i in range(len(labels)):
        for label in labels[i][1]:
            parse_image[np.where(ori_result_parse_image == label)] = i
    ##### get face mask from origin image atr parse
    face_mask = ((b_parse_image == 3).astype(np.float32) + (b_parse_image == 11).astype(np.float32))

    #### get neck(atr parse of face mask - lip parse of face mask) mask from viton result parse
    target_mask =((parse_image == 2).astype(np.float32) - (face_mask == 1).astype(np.float32))
    im = result_raw_image.copy()  #human img PIL
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    im = Image.fromarray(img)

    white = Image.new("RGB", (768, 1024), (255, 255, 255)) #white img PIL

    white.paste(im, None, Image.fromarray(np.uint8(target_mask * 255), 'L'))

    #open and close(require cv2)
    white_cv2 = cv2.cvtColor(np.asarray(white), cv2.COLOR_RGB2BGR) #pic white: PIL to cv2

    kernel = np.ones((10, 10), np.uint8)  # 获取矩形结构元
    new_neck = cv2.morphologyEx(white_cv2, cv2.MORPH_CLOSE, kernel, iterations=5)  #neck with better shape

    #cv2 smooth filter
    # dst_f = cv2.blur(dst, (5, 5))

    #add neck mask to full_body img
    fullbody = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    rows, cols, channels = new_neck.shape
    roi = fullbody[0:rows, 0:cols]

    new_neckgray = cv2.cvtColor(new_neck, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(new_neckgray, 100, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)  #

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

    img2_fg = cv2.bitwise_and(new_neck, new_neck, mask=mask_inv)

    dst = cv2.add(img1_bg, img2_fg)
    fullbody[0:rows, 0:cols] = dst
    # cv2.imshow('res', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows
    return dst



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
    pose_name = 'cd60a1700f868b80e5e0b695043576d'
    result_list = os.listdir('/home/vera/myCode/top_pose_process_examples/smooth_test1/viton_result')
    result_raw_image_dir = '/home/vera/myCode/top_pose_process_examples/smooth_test1/viton_result/'
    result_parse_image_dir = '/home/vera/myCode/top_pose_process_examples/smooth_test1/viton_result_parse_atr/'
    origin_c_image_path = '/home/vera/myCode/top_pose_process_examples/smooth_test1/image/{}.jpg'.format(pose_name)
    b_parse_image_path = '/home/vera/myCode/top_pose_process_examples/smooth_test1/image-parse-v3-atr/{}.png'.format(pose_name)
    agnostic_mask_path = '/home/vera/myCode/top_pose_process_examples/smooth_test1/agnostic_mask/{}.png'.format(pose_name)
    input_h = 1024

    origin_c_image = cv2.imread(origin_c_image_path)
    b_parse_image = Image.open(b_parse_image_path)
    b_parse_image = np.asarray(b_parse_image)
    agnostic_mask = Image.open(agnostic_mask_path)
    agnostic_mask = np.asarray(agnostic_mask)

    des_dir = '/home/vera/myCode/top_pose_process_examples/smooth_test1/output_smooth1'


    if not os.path.exists(des_dir):
        os.makedirs(des_dir)


    for i in range(len(result_list)):

        result_raw_image = cv2.imread(result_raw_image_dir + result_list[i] )

        ori_result_parse_image = Image.open(result_parse_image_dir + result_list[i])
        ori_result_parse_image = np.asarray(ori_result_parse_image)
        result = post_process_viton_top_image(result_raw_image,origin_c_image,b_parse_image,agnostic_mask,ori_result_parse_image,top_labels,input_h=1024)
        cv2.imwrite(os.path.join(des_dir, result_list[i]), result)