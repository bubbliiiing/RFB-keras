#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
from nets.rfb import rfb300

if __name__ == "__main__":
    NUM_CLASSES = 21
    input_shape = (300, 300, 3)

    model = rfb300(input_shape, num_classes=NUM_CLASSES)
    model.summary()
    