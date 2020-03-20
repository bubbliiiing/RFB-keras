from keras.layers import Input
from rfb import RFB
from PIL import Image

rfb = RFB()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = rfb.detect_image(image)
        r_image.show()
rfb.close_session()
    