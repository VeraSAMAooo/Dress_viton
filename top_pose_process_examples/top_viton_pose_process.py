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

smooth_mask = SoftErosion(kernel_size=3, threshold=0.9, iterations=1).cuda()

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
                                 ori_result_parse_image,labels,input_h=1024,need_full_png=False,
                                 f_image=None,body_parse=None,bbox=None):

    #### ori_parse_image is the atr parse of the viton result
    #### map 18 atr labels to 13 parse labels
    ori_result_parse_image = np.asarray(ori_result_parse_image)
    parse_image = np.zeros((ori_result_parse_image.shape[0], ori_result_parse_image.shape[1]))
    for i in range(len(labels)):
        for label in labels[i][1]:
            parse_image[np.where(ori_result_parse_image == label)] = i

    ##### get face mask from origin image atr parse(already processed face mask with lip parse result )
    face_mask = ((b_parse_image == 3).astype(np.float32) + (b_parse_image == 11).astype(np.float32))

    #### get upper, left_arm,right_arm, neck(atr parse of face mask - lip parse of face mask) mask from viton result parse
    target_mask = ((parse_image == 3).astype(np.float32) +
                   (parse_image == 5).astype(np.float32) +
                   (parse_image == 6).astype(np.float32) +
                   (parse_image == 2).astype(np.float32) -
                   (face_mask == 1).astype(np.float32)) * 255

    ##### dilate face_mask in order to make the face_mask cover more area of neck
    ##### can adjust kernel size
    face_mask = face_mask * 255
    kernel = np.ones((round(15*input_h/1024), round(15*input_h/1024)), np.uint8)
    face_mask = cv2.dilate(face_mask, kernel, iterations=1)
    ##### set coordinate to 0 where dilated face_mask has value
    target_mask[np.where(face_mask > 0)] = 0

    #### get upper and arm mask from input image atr parse(already processed face mask with lip parse result )
    b_upper_coor_list = list()
    b_arm_coor_list = list()
    for label in [4, 7]:
        b_upper_coor_list.append(np.where(b_parse_image == label))

    for label in [14, 15]:
        b_arm_coor_list.append(np.where(b_parse_image == label))
    for coor in b_upper_coor_list:
        target_mask[coor] = 255
    for coor in b_arm_coor_list:
        target_mask[coor] = 255

    #### remove pixels not in angositc area
    #### in case the parse result of viton result maybe not so good, e.g some pixel in bottom  will parsed as upper
    target_mask[np.where(agnostic_mask < 10)] = 0

    #### blur target mask around neck and face skin area
    #### can adjust erode_pixels and kernal_size
    coor_raw = np.where(target_mask > 0)
    min_y, max_y = min(coor_raw[0]), max(coor_raw[0])
    erode_pixels = round(20*input_h/1024)
    p_h ,p_w = parse_image.shape[:2]
    erode_mask = np.zeros((p_h ,p_w))
    erode_mask[min_y:min_y +erode_pixels] = target_mask[min_y:min_y +erode_pixels]
    kernel = np.ones((5 ,5) ,np.uint8)
    erode_mask = cv2.dilate(erode_mask ,kernel ,iterations = 1)
    kernel_size = (round(15*input_h/1024), round(15*input_h/1024))
    blur_size = tuple( 2 * i +1 for i in kernel_size)
    erode_mask = cv2.GaussianBlur(erode_mask, blur_size, 0)
    target_mask[:min_y +erode_pixels -5] = erode_mask[:min_y +erode_pixels -5]


    #### merge viton result and input image with target_mask smoothly
    target_image_parsing = postprocess(result_raw_image, origin_c_image,
                                             target_mask, smooth_mask)
    result = target_image_parsing[:, :, ::-1]

    # for png mask generation
    if need_full_png:
        b_w, b_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        c_h, c_w = result_raw_image.shape[:2]

        #### get upper, left_arm, right_arm, face mask(include neck) from viton result parse
        c_r_body_mask = np.zeros((c_h, c_w))
        for label in [3, 5, 6, 2]:
            c_r_body_mask[np.where(parse_image == label)] = 255

        kernel = np.ones((5, 5), np.uint8)
        erode_agnostic_mask = cv2.erode(agnostic_mask, kernel, iterations=1)

        #### remove pixels not in angositc area,
        #### in case the parse result of viton result maybe not so good, e.g, parse the top and skirt together to dress
        c_r_body_mask[np.where(erode_agnostic_mask <= 10)] = 0

        #### get cropped origin full body parse
        c_body_parse = body_parse[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        #### get resized cropped origin full body parse
        if b_h > c_h:
            c_r_body_parse = cv2.resize(c_body_parse, (c_w, c_h), cv2.INTER_LINEAR)
        else:
            c_r_body_parse = cv2.resize(c_body_parse, (c_w, c_h), cv2.INTER_CUBIC)

        #### only keep the mask that not in agnosic_mask area
        c_r_body_parse[np.where(erode_agnostic_mask > 10)] = 0

        #### add mask from viton result rsult and origin body parse
        c_body_mask = c_r_body_parse.reshape((c_h, c_w, 1)) + c_r_body_mask.reshape((c_h, c_w, 1))
        c_body_mask = np.clip(c_body_mask ,0 ,255)

        if b_h > c_h:
            result = cv2.resize(result, (b_w, b_h), cv2.INTER_CUBIC)
            c_body_mask = cv2.resize(c_body_mask, (b_w, b_h), cv2.INTER_CUBIC)
        else:
            result = cv2.resize(result, (b_w, b_h), cv2.INTER_AREA)
            c_body_mask = cv2.resize(c_body_mask, (b_w, b_h), cv2.INTER_AREA)

        h ,w = f_image.shape[:2]
        f_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = result
        body_parse[bbox[1]:bbox[3], bbox[0]:bbox[2]] = c_body_mask.reshape(b_h, b_w)

        #### remove small disconnected pixels
        kernel = np.ones((round(5*input_h/1024), round(5*input_h/1024)), np.uint8)
        body_parse = cv2.erode(body_parse, kernel, iterations=1)
        kernel = np.ones((round(5*input_h/1024),round(5*input_h/1024)), np.uint8)
        body_parse = cv2.dilate(body_parse, kernel, iterations=1)
        result =  np.concatenate([f_image ,body_parse.reshape(h ,w ,1)] ,axis=2)

    return result


if __name__ == '__main__':
    import glob
    import os
    import json
    top_labels = {
        0: ['background', [0]],
        1: ['hair', [1, 2]],
        2: ['face', [3, 11]],
        3: ['upper', [4, 7]],
        4: ['bottom', [5, 6, 8]],
        5: ['left_arm', [14]],
        6: ['right_arm', [15]],
        7: ['left_leg', [12]],
        8: ['right_leg', [13]],
        9: ['left_shoe', [9]],
        10: ['right_shoe', [10]],
        11: ['socks', []],
        12: ['noise', [16, 17]]
    }
    pose_name = '6647234_23901965_3600_2400'
    result_raw_image_dir = './top_viton_post_process/viton_result'
    result_parse_image_dir = './top_viton_post_process/viton_result_parse_atr'
    origin_c_image_path = './top_viton_post_process/image/{}.jpg'.format(pose_name)
    b_parse_image_path = './top_viton_post_process/image-parse-v3-atr/{}.png'.format(pose_name)
    agnostic_mask_path = './top_viton_post_process/agnostic_mask/{}.png'.format(pose_name)

    input_h = 1024
    f_image_path = './top_viton_post_process/full_image/{}.jpg'.format(pose_name)
    body_parse_path = './top_viton_post_process/body_parse/{}.png'.format(pose_name)
    bbox_file_path = './top_viton_post_process/kp_bbox_map.json'
    origin_c_image = cv2.imread(origin_c_image_path)
    b_parse_image = Image.open(b_parse_image_path)
    b_parse_image = np.asarray(b_parse_image)
    agnostic_mask = Image.open(agnostic_mask_path)
    agnostic_mask = np.asarray(agnostic_mask)
    full_image = cv2.imread(f_image_path)
    body_parse = cv2.imread(body_parse_path,cv2.IMREAD_UNCHANGED)
    with open(bbox_file_path,'r') as file:
        bbox_map = json.loads(file.read())
    bbox = bbox_map.get('{}.jpg'.format(pose_name)).get('bbox')
    des_dir = './top_viton_post_process/output1'
    des_png_dir = './top_viton_post_process/output_png1'

    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    if not os.path.exists(des_png_dir):
        os.makedirs(des_png_dir)

    for result_raw_image_path in glob.glob(os.path.join(result_raw_image_dir,'{}*'.format(pose_name))):
        print(result_raw_image_path)
        name = result_raw_image_path.strip().split('/')[-1].split('.')[0]
        result_raw_image = cv2.imread(result_raw_image_path)
        result_parse_image_path = os.path.join(result_parse_image_dir,'{}.png'.format(name))
        ori_result_parse_image = Image.open(result_parse_image_path)
        ori_result_parse_image = np.asarray(ori_result_parse_image)
        #### get result
        result = post_process_viton_top_image(result_raw_image,origin_c_image,b_parse_image,agnostic_mask,
                                 ori_result_parse_image,top_labels,input_h=1024,need_full_png=False,
                                 f_image=None,body_parse=None,bbox=None)
        cv2.imwrite(os.path.join(des_dir,'{}.png'.format(name)),result)
        #### get full png result
        result = post_process_viton_top_image(result_raw_image,origin_c_image,b_parse_image,agnostic_mask,
                                 ori_result_parse_image,top_labels,input_h=1024,need_full_png=True,
                                 f_image=full_image,body_parse=body_parse,bbox=bbox)
        cv2.imwrite(os.path.join(des_png_dir,'{}.png'.format(name)),result)


