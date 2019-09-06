#coding=utf-8
from __future__ import division,print_function
import itertools
import math
import os
import random
import sys
import numpy as np
import cv2
import codecs
import json
# import pdb

from img_utils import *
from jittering_methods import *
from parse_args import parse_args

args = parse_args()
fake_resource_dir  = sys.path[0] + "/fake_resource/" 
output_dir = args.img_dir
resample_range = args.resample 
gaussian_range = args.gaussian 
noise_range = args.noise
rank_blur = args.rank_blur
brightness = args.brightness
motion_blur = args.motion_blur
number_dir = [fake_resource_dir + "/numbers/",fake_resource_dir + "/numbers1/",fake_resource_dir + "/numbers2/" , fake_resource_dir + "/numbers3/",fake_resource_dir + "/numbers4/"]
letter_dir = [fake_resource_dir + "/letters/" ,fake_resource_dir + "/letters1/",fake_resource_dir + "/letters2/" , fake_resource_dir + "/letters3/",fake_resource_dir + "/letters4/"]
plate_dir = [fake_resource_dir + "/plate_background_use/" ,fake_resource_dir + "/plate_background_use1/"]
screw_dir = [fake_resource_dir + "/screw/",fake_resource_dir + "/screw1/"]


# character_y_size = 100
character_y_size = 110
plate_y_size = 150
# plate_y_size = 164

class FakePlateGenerator():
    def __init__(self, plate_size):
        font = random.randint(0,4)
        color = random.randint(0,1)
        self.dst_size = plate_size

        #self.chinese = self.load_image(chinese_dir, character_y_size)
        self.numbers = self.load_image(number_dir[font], character_y_size)
        self.letters = self.load_image(letter_dir[font], character_y_size)

        self.numbers_and_letters = dict(self.numbers, **self.letters)

        #we only use blue plate here
        self.plates, self.plate_x_size = self.load_plate_image(plate_dir[color], plate_y_size)
        self.screws = self.load_screws(screw_dir[color],plate_y_size)
    
        for i in self.plates.keys():
            self.plates[i] = cv2.cvtColor(self.plates[i], cv2.COLOR_BGR2BGRA)

        #positions 
        self.character_position_x_listStart = [60+30,90+30, 120+30,150+30]
        self.character_position_x_listRest = [] 
    
    def get_radom_sample(self, data):
        keys = list(data.keys())
        i = random.randint(0, len(data) - 1)
        key = keys[i]
        value = data[key]

        #注意对矩阵的深拷贝
        return key, value.copy()

    def load_image(self, path, dst_y_size):
        img_list = {}
        current_path = sys.path[0]

        listfile = os.listdir(path)     
        for filename in listfile:
            img = cv2.imread(path + filename, -1)
            
            height, width = img.shape[:2]
            x_size = int(width*(dst_y_size/float(height)))
            img_scaled = cv2.resize(img, (x_size, dst_y_size), interpolation = cv2.INTER_CUBIC)
            
            img_list[filename[:-4]] = img_scaled

        return img_list
    
    def load_plate_image(self, path, dst_y_size):
        img_list = {}
        current_path = sys.path[0]

        listfile = os.listdir(path)     

        for filename in listfile:
            img = cv2.imread(path + filename, -1)
            
            height, width = img.shape[:2]
            x_size = int(width*(dst_y_size/float(height)))+50
            img_scaled = cv2.resize(img, (x_size, dst_y_size), interpolation = cv2.INTER_CUBIC)
            
            img_list[filename[:-4]] = img_scaled

        return img_list, x_size
    
    def load_screws(self, path, dst_y_size):
        img_list = {}
        current_path = sys.path[0]

        listfile = os.listdir(path)     

        for filename in listfile:
            img = cv2.imread(path + filename, -1)
            img_list[filename[:-4]] = img

        return img_list

    def add_character_to_plate(self, character, plate, x):
        h_plate, w_plate = plate.shape[:2]
        h_character, w_character = character.shape[:2]

        start_x = x - int(w_character/2)
        start_y = int((h_plate - h_character)/2)

        a_channel = cv2.split(character)[3]
        ret, mask = cv2.threshold(a_channel, 100, 255, cv2.THRESH_BINARY)
        character = emboss(character)
        overlay_img(character, plate, mask, start_x, start_y)

        return start_x, start_y
    
    def add_screws_to_plate(self, character, plate, x):
        h_plate, w_plate = plate.shape[:2]
        h_character, w_character = character.shape[:2]

        start_x = x - int(w_character/2)
        start_y = 10

        a_channel = cv2.split(character)[3]
        ret, mask = cv2.threshold(a_channel, 100, 255, cv2.THRESH_BINARY)
        overlay_img(character, plate, mask, start_x, start_y)

    def generate_one_plate(self):
        plate_chars = ""
        _, plate_img = self.get_radom_sample(self.plates)
        plate_name = ""

        num = random.randint(3, 30)#6
        num = 6 if num >= 6 else num    
        start_xy= []

        # i = (len(self.character_position_x_list) - num)//2 - 1
        i = 6 - num
        #spacing = random.randint(55,65) #60 for normal spacing
        character, img1 = self.get_radom_sample(self.letters)
        start_xy.append(self.add_character_to_plate(img1, plate_img, self.character_position_x_listStart[i]))
        plate_name += "%s"%(character,)
        plate_chars += character

        character, img = self.get_radom_sample(self.letters)
        start_xy.append(self.add_character_to_plate(img, plate_img, self.character_position_x_listStart[i]+60))
        plate_name += "%s"%(character,)
        plate_chars += character

        self.character_position_x_listRest = [] 
        for j in range(2,7):
            self.character_position_x_listRest.append(self.character_position_x_listStart[i] + (j*60))
        self.character_position_x_listRest = [x.__sub__(20) for x in self.character_position_x_listRest]

        #makes sure first digit does not start with a 0
        while True:
            character, img2 =  self.get_radom_sample(self.numbers)
            if int(character) != 0:
                start_xy.append(self.add_character_to_plate(img2, plate_img, self.character_position_x_listRest[1]))
                plate_name += character
                plate_chars += character
                break

        for j in range(4,num+1):
            character, img2 =  self.get_radom_sample(self.numbers_and_letters)
            start_xy.append(self.add_character_to_plate(img2 ,plate_img, self.character_position_x_listRest[j-2]))
            plate_name += character
            plate_chars += character
        screw, img = self.get_radom_sample(self.screws)
        self.add_screws_to_plate(img, plate_img, 110+20)
        self.add_screws_to_plate(img, plate_img, 350+20)

        #转换为RBG三通道
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGRA2BGR)
  
        #转换到目标大小
        pt1 = (start_xy[0][0],start_xy[0][1])
        pt2 = (start_xy[num+1 - 2][0] + img2.shape[1], start_xy[num+1 - 2][1] + character_y_size)

        plate_img = cv2.resize(plate_img, self.dst_size, interpolation = cv2.INTER_AREA)

        scalex = self.dst_size[0]/self.plate_x_size
        scaley = self.dst_size[1]/plate_y_size
        pts1 = (int(pt1[0] * scalex)-3,int(pt1[1] * scaley)-3) 
        pts2 = (int(pt2[0] * scalex)+3,int(pt2[1] * scaley)+3)

        # cv2.rectangle(plate_img, pts1,pts2, (255,0,0), 2)
        # cv2.imshow(' ', plate_img)
        # cv2.waitKey(0)
        box = (pts1[0],pts1[1], pts2[0] - pts1[0], pts2[1] - pts1[1])
        return plate_img, plate_name, plate_chars, box

def write_to_txt(fo,img_name, plate_characters):
    plate_label = '|' + '|'.join(plate_characters) + '|'
    line = img_name + ';' + plate_label.upper() + '\n'
    print(line.encode('utf8'))
    fo.write("%s" % line)

def json_generator(json_file, fname, plate_chars,box,json_data):
    i=0
    plate_anno =[]
    plate_anno.append(
        {"box": box,
        "text": plate_chars,}
    )
    if fname not in json_data:
        json_data[fname.split('/')[len(fname.split('/'))-1]] = plate_anno


if __name__ == "__main__":
    # fake_resource_dir  = sys.path[0] + "/fake_resource/" 
    # output_dir = sys.path[0] + "/test_plate/"
    json_data = {}
    img_size = (300, 90)#100,30
    json_file = output_dir + "/ocr_label.json"
    reset_folder(output_dir)
    numImgs = args.num_imgs
    fo = codecs.open(output_dir + 'labels.txt', "w", encoding='utf-8')
    for i in range(0, numImgs):
        if i % 100 == 0:
            print (i)
        fake_plate_generator = FakePlateGenerator( img_size)
        plate, plate_name, plate_chars,box = fake_plate_generator.generate_one_plate()
        plate = underline(plate)
        plate = jittering_color(plate)
        plate = add_noise(plate,noise_range)
        plate = jittering_blur(plate,gaussian_range)
        plate = resample(plate, resample_range)
        plate = jittering_scale(plate)
        #plate = invertColor(plate)
        # plate = perspectiveTransform(plate)
        plate = random_rank_blur(plate,rank_blur)
        plate = random_motion_blur(plate,motion_blur)
        plate = random_brightness(plate, brightness)
        file_name = save_random_img(output_dir,plate_chars.upper(), plate)
        write_to_txt(fo,file_name,plate_chars)
        json_generator(json_file, file_name,plate_chars.upper(), box,json_data)
    with codecs.open(json_file, 'w') as f:
        f.write(json.dumps(json_data,indent=4))
