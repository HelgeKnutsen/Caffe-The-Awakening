import pygame.camera
import time
import collections

from camera import startCamera, takePicture, dispPicture

caffe_init = caffeSetup()
labels, transformer, net = caffe_init

cam = startCamera()

while (1):

    image = takePicture(cam)
    dispPicture(image.disp)

    prob = classify(image.classify, transformer, net)

    printClassPred(prob, labels)

    time.sleep(0.01)

cam.stop()