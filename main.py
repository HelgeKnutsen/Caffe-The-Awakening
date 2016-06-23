import pygame.camera
import time
import collections

from camera import startCamera, takePicture, dispPicture

cam = startCamera()

while (1):

    image = takePicture(cam)
    dispPicture(image.disp)

    #classify(image.classify)
    time.sleep(0.01)

cam.stop()