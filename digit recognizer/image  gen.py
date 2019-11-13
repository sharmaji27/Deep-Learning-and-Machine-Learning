from PIL import ImageGrab
import time

images_folder = "training data/9/"

for i in range(0, 45):
    time.sleep(3)
    im = ImageGrab.grab(bbox=(10, 230, 260, 476))  # X1,Y1,X2,Y2
    print("saved....", i)
    im.save(images_folder + str(i) + '.png')
    print("clear screen now and redraw now...")