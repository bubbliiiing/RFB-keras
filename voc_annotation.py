import xml.etree.ElementTree as ET
from os import getcwd
import numpy as np


sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

nums = np.zeros(len(classes))
def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1
wd = getcwd()

def yes_or_no(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC2007/Annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    flag = False
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        flag = True
    return flag

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        in_file = open('VOCdevkit/VOC2007/Annotations/%s.xml'%(image_id))
        tree=ET.parse(in_file)
        root = tree.getroot()
        if not yes_or_no(year, image_id, list_file):
            continue
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()

for i in range(len(classes)):
    print(nums[i],classes[i])