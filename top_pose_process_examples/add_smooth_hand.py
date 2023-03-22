import cv2
from PIL import Image
import numpy as np
from copy import deepcopy
import torch
from torch.nn import functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

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

def post_process_viton_top_image(result_raw_image,origin_c_image,target_mask, input_h=1024):
    #### ori_parse_image is the atr parse of the viton result
    #### map 18 atr labels to 13 parse labels


    #### blur target mask around neck and face skin area
    #### can adjust erode_pixels and kernal_size
    # coor_raw = np.where(target_mask > 0)
    # min_y, max_y = min(
    #     coor_raw[0]), max(coor_raw[0])
    # erode_pixels = round(10*input_h/512)
    #
    # erode_mask = np.zeros((1024 , 768))
    # erode_mask[min_y:min_y +erode_pixels] = target_mask[min_y:min_y +erode_pixels]
    # kernel = np.ones((5 ,5) ,np.uint8)
    # erode_mask = cv2.dilate(erode_mask ,kernel ,iterations = 1)
    # kernel_size = (round(10*input_h/1024), round(10*input_h/1024))
    # blur_size = tuple( 10 * i +1 for i in kernel_size)
    # erode_mask = cv2.GaussianBlur(erode_mask, blur_size, 0)
    # target_mask[:min_y +erode_pixels -5] = erode_mask[:min_y +erode_pixels -5]

    #### merge viton result and input image with target_mask smoothly
    target_image_parsing = postprocess(result_raw_image, origin_c_image,
                                             target_mask, smooth_mask)


    result = target_image_parsing[:, :, ::-1]

    return result

result_raw_image= Image.open('/home/vera/myCode/top_pose_process_examples/set_0309/hand_test/01.png_2334941_out.png')
arm1 = cv2.imread('/home/vera/myCode/top_pose_process_examples/set_0309/arm/Lucy_arm.png',cv2.IMREAD_UNCHANGED)
arm_mask = arm1[:,:,3]
img_jpg = Image.open('/home/vera/myCode/top_pose_process_examples/set_0309/arm/Lucy_arm.png')
img_jpg = np.asarray(img_jpg)[:,:,:3]

result = post_process_viton_top_image(img_jpg,result_raw_image, arm_mask, input_h=1024)
cv2.imwrite("/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0309/cc.jpg", result[:,:,::-1])