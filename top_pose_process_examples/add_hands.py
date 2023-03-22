import cv2
from PIL import Image
import numpy as np
import os

# full_body = Image.open('/home/vera/myCode/top_pose_process_examples/set_0309/hand_test/01.png_2334941_out.png')
# arm = Image.open('/home/vera/myCode/top_pose_process_examples/set_0309/arm/Lucy_arm.png', cv2.IMREAD_UNCHANGED)
# img_jpg = arm[:,:,:3]

arm1 = cv2.imread('/home/vera/myCode/top_pose_process_examples/set_0309/arm/Lucy_arm_only_hand_2.png',cv2.IMREAD_UNCHANGED)
img_jpg = Image.open('/home/vera/myCode/top_pose_process_examples/set_0309/arm/Lucy_arm_only_hand_2.png')
arm_mask = arm1[:,:,3]


# full_body.paste(img_jpg, None, Image.fromarray(np.uint8(arm_mask), 'L'))
#
# full_body.show()

full_body_list = os.listdir('/home/vera/myCode/top_pose_process_examples/set_0309/output06')
full_body_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/output06/'

des_dir = '/home/vera/myCode/top_pose_process_examples/set_0309/output06_hand/'

if not os.path.exists(des_dir):
    os.makedirs(des_dir)

for i in range(len(full_body_list)):
    fullbody_image = Image.open(full_body_dir + full_body_list[i])
    img_jpg1 = img_jpg.copy()
    arm_mask1 = arm_mask.copy()
    fullbody_image.paste(img_jpg1, None, Image.fromarray(np.uint8(arm_mask1), 'L'))

    result = fullbody_image.save(os.path.join(des_dir, full_body_list[i]))
