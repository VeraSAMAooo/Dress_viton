from PIL import Image
import numpy as np
import os
import cv2

def parse_fullbody(background, fullbody, mask):
    background.paste(fullbody, None, Image.fromarray(np.uint8(mask), 'L'))
    return background

if __name__ == '__main__':
    fullbody_list = os.listdir('/home/vera/myCode/top_pose_process_examples/set_0309/all_res/olivia/olivia-res-body/olivia-color-bgno')
    fullbody_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/all_res/olivia/olivia-res-body/olivia-color-bgno/'
    # mask_dir = '/media/Algorithm/vera/dress_body_parse_20230315/green-res-body/green-sleeveless/'

    des_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/all_res/olivia/olivia-res-body/olivia-color-bgno-gray/'
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    img1 = cv2.imread('/home/vera/myCode/top_pose_process_examples/set_0309/all_res/olivia/olivia-res-body/olivia-color-bgno/Olivia_midi.png_5436599_out.png')
    img2 = img1[:,:,:3]
    cv2.imwrite('/home/vera/myCode/top_pose_process_examples/set_0309/all_res/olivia/olivia-res-body/olivia-color-bgno/Olivia_1.png', img2)


    # for i in range(len(fullbody_list)):
    #     try:
    #         fullbody_image = Image.open(fullbody_dir + fullbody_list[i])
    #         width = fullbody_image.width
    #         height = fullbody_image.height
    #         # if fullbody_image.mode != 'RGBA':
    #         #     fullbody_image = fullbody_image.convert('RGBA')
    #
    #         # mask_image = Image.open(mask_dir + fullbody_list[i])
    #         gray_back = Image.new("RGB", size = (width, height), color = (128, 128, 128))
    #
    #
    #
    #         background = gray_back.copy()
    #         background.paste(fullbody_image, None, mask = fullbody_image)
    #
    #         # background.paste(fullbody_image, None, Image.fromarray(np.uint8(mask_image), 'L'))
    #
    #         result = background.save(os.path.join(des_dir, fullbody_list[i]))
    #     except:
    #         print('no')
