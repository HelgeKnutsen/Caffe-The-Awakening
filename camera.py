import pygame.camera
import collections
from selectivesearch import selective_search
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

resX = 1920
resY = 1080

def startCamera():
    pygame.camera.init()
    pygame.camera.list_cameras()
    cam = pygame.camera.Camera("/dev/video1", (resX, resY))
    cam.start()
    return cam


def takePicture(cam):

    # Take picture, camPic is Surface type
    im_s = cam.get_image()

    # Convert image to numpy array type
    im_n = pygame.surfarray.array3d(im_s)/255.0

    # Create a tuple that returns both types. We need the surface type to display to screen and the numpy array type to classify with Caffe
    imageTypes = collections.namedtuple('imageTypes', ['disp', 'classify'])

    image = imageTypes(im_s, im_n)

    return image


# camPic needs to be of Surface type
def dispPicture(im_s):

    # Initialize image size on screen
    screen = pygame.display.set_mode((resX, resY))

    # Show image on screen
    screen.blit(im_s, (0, 0))
    pygame.display.flip()

    return

def selSearch(im_n):

    print im_n

    img_lbl, regions = selective_search(im_n, scale = 1000, sigma = 0.9, min_size = 3000)

    candidates = set()
    for r in regions:

        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            #print 'Already done'
            continue

        # excluding regions smaller than 2000 pixels
        # if r['size'] < 2000:
            #print 'Too small'
            #continue

        # distorted rects
        x, y, w, h = r['rect']
        if w == 0 or h == 0 or w / h > 1.2 or h / w > 1.2:
            #print 'Distorted'
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
    ax.imshow(im_n)
    for x, y, w, h in candidates:
        print x, y, w, h
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.draw()

    return candidates
