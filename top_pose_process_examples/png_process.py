from PIL import Image
import cv2

import numpy as np

img1 = cv2.imread('/home/vera/myCode/top_pose_process_examples/set2/model_bkn/Avery_Linen_Shorts_Oat_The_Willow_Label_01.png',cv2.IMREAD_UNCHANGED)
# img1.show()

def transparence2white(img):
    shape = img.shape
    width = shape[0]
    height = shape[1]
    mask = img[:,:,3]
    img_jpg = img[:,:,:3]
    img_jpg[np.where(mask < 50)] = 255
    # for yh in range(height):
    #     for xw in range(width):
    #         color_d = img[xw, yh]
    #         if(color_d.size != 4):
    #             continue
    #         if(color_d[3] == 0):
    #             img[xw, yh] == [255, 255, 255, 255]  #make the pixel white

    return img_jpg

img1 = transparence2white(img1)

# cv2.imshow("1.jpg", img1)

# cv2.waitKey(0)

cv2.imwrite('/home/vera/myCode/top_pose_process_examples/set2/model_bkn/Avery_Linen_Shorts_Oat_The_Willow_Label_01.jpg', img1)