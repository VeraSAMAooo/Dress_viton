import numpy as np
import cv2
from PIL import Image
import os

def changeColor():
    img_list = os.listdir('/home/vera/myCode/top_pose_process_examples/set_0309/gray')
    img_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/gray/'
    res_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/gray/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    for i in range(len(img_list)):

        image = cv2.imread(img_dir + img_list[i])
        # image in this case is your image you want to eliminate black
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out = 255 - gray
        cv2.imwrite(res_dir + img_list[i].replace('.jpg', '.png'), out)

def resize():
    result_list = os.listdir('/home/vera/myCode/top_pose_process_examples/set_0309/temp_fullbody0/generator_res_1.5')
    result_parse_image_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/temp_fullbody0/generator_res_1.5/'
    res_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/temp_fullbody0/generator_res-resize1024/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    for i in range(len(result_list)):

        ori = Image.open(result_parse_image_dir + result_list[i])
        newsize = (768, 1024)
        ori = ori.resize(newsize)
        ori.save(res_dir + result_list[i].replace('.jpg', '.png'))


def j2p():
    img_list = os.listdir('/home/vera/myCode/top_pose_process_examples/set2/viton_result')
    img_dir = '/home/vera/myCode/top_pose_process_examples/set2/viton_result/'
    res_dir = '/home/vera/myCode/top_pose_process_examples/set2/viton_result-res/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    for i in range(len(img_list)):

        image = cv2.imread(img_dir + img_list[i])
        cv2.imwrite(res_dir + img_list[i].replace('.jpg', '.png'), image)



if __name__ == '__main__':
    # changeColor()
    resize()
    # j2p()


    # #random select list
    # image = cv2.imread('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0309/image/01.png')
    # # image in this case is your image you want to eliminate black
    # # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # out = 255 - gray
    # cv2.imwrite('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0309/image/01.jpg', image)

    # gray_back = Image.new("RGB", (768, 1024), (128, 128, 128))
    # gray_back.save('/home/vera/myCode/top_pose_process_examples/set_0309/all_res/olivia/olivia-res-parse/olivia-shortsleeve/Olivia_midi.png_7543172_out.png')