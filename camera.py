import pygame.camera
import collections
from scipy import misc
import subprocess
import os

resX = 1920
resY = 1080

def startCamera():
    pygame.camera.init()
    pygame.camera.list_cameras()
    cam = pygame.camera.Camera("/dev/video0", (resX, resY))
    cam.start()
    return cam

def loadImage(path, imName):

    imFullQ = misc.imread(path + imName)

    #TEST DIFFERENT INTERPOLATION TECHNIQUES: 'nearest', 'bilinear', 'bicubic', 'cubic'
    #imLowQ = misc.imresize(imFullQ, 0.2, 'cubic')

    #imageQ = collections.namedtuple('imageQ', ['highQ', 'lowQ'])

    #image = imageQ(imFullQ, imLowQ)

    return imFullQ

def takePicture(cam):

    # Take picture, camPic is Surface type
    im_s = cam.get_image()

    # Convert image to numpy array type
    im_n = pygame.surfarray.array3d(im_s)/255.0

    # Create a tuple that returns both types. We need the surface type to display to screen and the numpy array type to classify with Caffe
    imageTypes = collections.namedtuple('imageTypes', ['disp', 'classify'])

    image = imageTypes(im_s, im_n)

    #plt.imshow(im_n, interpolation='nearest')
    #plt.draw()

    return image


# camPic needs to be of Surface type
def dispPicture(im_s):

    # Initialize image size on screen
    screen = pygame.display.set_mode((resX, resY))

    # Show image on screen
    screen.blit(im_s, (0, 0))
    pygame.display.flip()

    return