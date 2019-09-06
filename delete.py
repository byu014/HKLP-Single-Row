import cv2 
import sys
import os

path = sys.path[0] + '/fake_resource/numbers/'
save_path = sys.path[0] + '/fake_resource/delete/'

for image_name in os.listdir(path):
    image = cv2.imread(path + image_name, -1)
    cv2.imwrite(save_path + image_name[:-3] + '.jpg', image)