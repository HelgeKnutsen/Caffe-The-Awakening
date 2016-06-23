import pygame.camera
import collections

def startCamera():
    pygame.camera.init()
    pygame.camera.list_cameras()
    cam = pygame.camera.Camera("/dev/video0", (1920, 1080))
    cam.start()
    return cam


def takePicture(cam):

    # Take picture, camPic is Surface type
    im_s = cam.get_image()

    # Convert image to numpy array type
    im_n = pygame.surfarray.array3d(im_s)

    # Create a tuple that returns both types. We need the surface type to display to screen and the numpy array type to classify with Caffe
    imageTypes = collections.namedtuple('imageTypes', ['disp', 'classify'])

    image = imageTypes(im_s, im_n)

    return image


# camPic needs to be of Surface type
def dispPicture(im_s):

    # Initialize image size on screen
    screen = pygame.display.set_mode((1920, 1080))

    # Show image on screen
    screen.blit(im_s, (0, 0))
    pygame.display.flip()

    return