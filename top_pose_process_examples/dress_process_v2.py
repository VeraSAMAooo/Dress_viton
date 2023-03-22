import cv2
from PIL import Image
import numpy as np
from copy import deepcopy
import torch
from torch.nn import functional as F
import torch.nn as nn

class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask

smooth_mask = SoftErosion(kernel_size=5, threshold=0.5, iterations=3).cuda()

def postprocess( swapped_face, target, target_mask, smooth_mask):
    face_mask_tensor = torch.from_numpy(target_mask.copy()).float().mul_(1 / 255.0).cuda()

    soft_face_mask_tensor, _ = smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
    soft_face_mask_tensor.squeeze_()

    soft_face_mask = soft_face_mask_tensor.cpu().numpy()
    soft_face_mask = soft_face_mask[:, :, np.newaxis]

    result = swapped_face * soft_face_mask + target * (1 - soft_face_mask)
    result = result[:, :, ::-1]  # .astype(np.uint8)
    return result

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
    target_mask = ((parse_image == 3).astype(np.float32) +
                   (parse_image == 4).astype(np.float32) +
                   (parse_image == 5).astype(np.float32) +
                   (parse_image == 6).astype(np.float32) +
                   (parse_image == 7).astype(np.float32) +
                   (parse_image == 2).astype(np.float32) -
                   (face_mask == 1).astype(np.float32)) * 255

    ##### dilate face_mask in order to make the face_mask cover more area of neck
    ##### can adjust kernel size
    face_mask = face_mask * 255
    kernel = np.ones((round(20*input_h/512), round(15*input_h/512)), np.uint8)
    face_mask = cv2.dilate(face_mask, kernel, iterations=1)
    ##### set coordinate to 0 where dilated face_mask has value
    target_mask[np.where(face_mask > 0)] = 0

    #### get upper and arm mask from input image atr parse(already processed face mask with lip parse result )
    b_upper_coor_list = list()
    b_arm_coor_list = list()
    b_leg_coor_list = list()
    for label in [4, 5, 6, 7, 8]:
        b_upper_coor_list.append(np.where(b_parse_image == label))

    for label in [14, 15, 12, 13]:
        b_arm_coor_list.append(np.where(b_parse_image == label))
    for coor in b_upper_coor_list:
        target_mask[coor] = 255
    for coor in b_arm_coor_list:
        target_mask[coor] = 255
    for coor in b_leg_coor_list:
        target_mask[coor] = 255

    #### remove pixels not in angositc area
    #### in case the parse result of viton result maybe not so good, e.g some pixel in bottom  will parsed as upper
    target_mask[np.where(agnostic_mask < 10)] = 0

    #### blur target mask around neck and face skin area
    #### can adjust erode_pixels and kernal_size
    coor_raw = np.where(target_mask > 0)
    min_y, max_y = min(
        coor_raw[0]), max(coor_raw[0])
    erode_pixels = round(10*input_h/512)
    p_h ,p_w = parse_image.shape[:2]
    erode_mask = np.zeros((p_h ,p_w))
    erode_mask[min_y:min_y +erode_pixels] = target_mask[min_y:min_y +erode_pixels]
    kernel = np.ones((5 ,5) ,np.uint8)
    erode_mask = cv2.dilate(erode_mask ,kernel ,iterations = 1)
    kernel_size = (round(10*input_h/1024), round(10*input_h/1024))
    blur_size = tuple( 10 * i +1 for i in kernel_size)
    erode_mask = cv2.GaussianBlur(erode_mask, blur_size, 0)
    target_mask[:min_y +erode_pixels -5] = erode_mask[:min_y +erode_pixels -5]

    #### merge viton result and input image with target_mask smoothly
    target_image_parsing = postprocess(result_raw_image, origin_c_image,
                                             target_mask, smooth_mask)


    result = target_image_parsing[:, :, ::-1]

    return result


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
    pose_name ='Lily'
    result_list = os.listdir('/home/vera/myCode/top_pose_process_examples/set_0309/viton_result')
    result_raw_image_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/viton_result/'
    result_parse_image_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/viton_result_parse_atr/'
    origin_c_image_path = '/home/vera/myCode/top_pose_process_examples/set_0309/image/{}.jpg'.format(pose_name)
    b_parse_image_path = '/home/vera/myCode/top_pose_process_examples/set_0309/image-parse-v3-atr/{}.png'.format(pose_name)
    agnostic_mask_path = '/home/vera/myCode/top_pose_process_examples/set_0309/agnostic_mask/{}.png'.format(pose_name)
    input_h = 1024

    origin_c_image = cv2.imread(origin_c_image_path)
    b_parse_image = Image.open(b_parse_image_path)
    b_parse_image = np.asarray(b_parse_image)
    agnostic_mask = Image.open(agnostic_mask_path)
    agnostic_mask = np.asarray(agnostic_mask)

    des_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/Lily-maxi1'


    if not os.path.exists(des_dir):
        os.makedirs(des_dir)


    for i in range(len(result_list)):

        result_raw_image = cv2.imread(result_raw_image_dir + result_list[i] )

        ori_result_parse_image = Image.open(result_parse_image_dir + result_list[i])
        ori_result_parse_image = np.asarray(ori_result_parse_image)
        result = post_process_viton_top_image(result_raw_image,origin_c_image,b_parse_image,agnostic_mask,ori_result_parse_image,top_labels,input_h=1024)

        #cv2 to pil
        img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        result = Image.fromarray(img)

        result_raw_image = cv2.cvtColor(result_raw_image, cv2.COLOR_BGR2RGB)
        result_raw_image = Image.fromarray(result_raw_image)

        #parse arms and legs on the result
        parse_array = np.array(ori_result_parse_image)
        parse_dress = ((parse_array == 7).astype(np.float32) +
                       (parse_array == 4).astype(np.float32) +
                       (parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 8).astype(np.float32))
        result.paste(result_raw_image, None, Image.fromarray(np.uint8(parse_dress * 255), 'L'))


        result = result.save(os.path.join(des_dir, result_list[i]))