import cv2
import numpy as np
import os
from PIL import Image


def merge_pics(dress, lily, olivia, green):


    hs_merge1 = np.concatenate((dress, lily), axis = 1)
    hs_merge2 = np.concatenate((olivia, green), axis = 1)

    res = np.concatenate((hs_merge1, hs_merge2), axis = 1)

    res = Image.fromarray(res)

    return res

def merge_2pics(pic1, pic2, pic3):


    hs_merge1 = np.concatenate((pic1, pic2, pic3), axis = 1)


    res = Image.fromarray(hs_merge1)

    return res
dress_list = os.listdir('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0306/dress/length/midi/cloth')
dress_img_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0306/dress/length/midi/cloth/'
dress_list.sort()

lily_list = os.listdir('/home/vera/myCode/top_pose_process_examples/set_0309/all_res/lily/lily-res-parse/lily-mini')
lily_img_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/all_res/lily/lily-res-parse/lily-mini/'
lily_list.sort()

olivia_list = os.listdir('/home/vera/myCode/top_pose_process_examples/set_0309/all_res/olivia/olivia-res-parse/olivia-mini')
olivia_img_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/all_res/olivia/olivia-res-parse/olivia-mini/'
olivia_list.sort()

green_list = os.listdir('/home/vera/myCode/top_pose_process_examples/set_0309/all_res/green/green-res-parse/green-mini')
green_img_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/all_res/green/green-res-parse/green-mini/'
green_list.sort()


# print(dress_list)
# des_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/all_res/merge_res/mini/'
# if not os.path.exists(des_dir):
#     os.makedirs(des_dir)
#
# for i in range(len(dress_list)):
#     dress_img = Image.open(dress_img_dir + dress_list[i])
#     # dress_img = dress_img.resize((384, 512))
#     dress_img = np.array(dress_img)
#
#     lily_img = Image.open(lily_img_dir + lily_list[i])
#     # lily_img = lily_img.resize((384, 512))
#     lily_img = np.array(lily_img)
#
#     olivia_img = Image.open(olivia_img_dir + olivia_list[i])
#     # olivia_img = olivia_img.resize((384, 512))
#     olivia_img = np.array(olivia_img)
#
#     green_img = Image.open(green_img_dir + green_list[i])
#     # green_img = green_img.resize((384, 512))
#     green_img = np.array(green_img)
#
#
#
#     result = merge_pics(dress_img, lily_img, olivia_img, green_img)
#     result.save((des_dir + dress_list[i]))
# #
#
# c1 = Image.open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0306/dress/sleeve_length/short_sleeve/cloth/7543172.jpg')
# c1 = c1.resize((384, 512))
# li = Image.open('/home/vera/myCode/top_pose_process_examples/set_0309/all_res/lily/lily-res-parse/lily-shortsleeve/Lily_midi.png_7543172_out.png')
# ol = Image.open('/home/vera/myCode/top_pose_process_examples/set_0309/all_res/olivia/olivia-res-body/olivia-shortsleeve/Olivia_midi.png_7543172_out.png')
# gr = Image.open('/home/vera/myCode/top_pose_process_examples/set_0309/all_res/green/green-res-parse/green-shortsleeve/midi1.png_7543172_out.png')
# result = merge_pics(c1, li, ol, gr)
# result.save("/home/vera/myCode/top_pose_process_examples/set_0309/all_res/four_merge/sleeve_length/shortsleeve/7543172.jpg")


pic1_list = os.listdir('/home/vera/myCode/top_pose_process_examples/set_0309/all_res/green/green-res-body/green-midi')
pic1_img_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/all_res/green/green-res-body/green-midi/'
pic1_list.sort()

pic2_list = os.listdir('/home/vera/myCode/top_pose_process_examples/set_0309/all_res/green/green-res-body/green-midi-1024')
pic2_img_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/all_res/green/green-res-body/green-midi-1024/'
pic2_list.sort()

des_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/all_res/green/green-res-body/compare/'
if not os.path.exists(des_dir):
    os.makedirs(des_dir)

for i in range(len(pic1_list)):
    pic1 = Image.open(dress_img_dir + dress_list[i])
    pic1 = np.array(pic1)
    pic2 = Image.open(pic1_img_dir + pic1_list[i])
    pic2 = np.array(pic2)
    pic3 = Image.open(pic2_img_dir + pic2_list[i])
    pic3 = np.array(pic3)

    res_merge = merge_2pics(pic1, pic2, pic3)
    res_merge.save((des_dir + dress_list[i]))