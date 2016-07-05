from selectivesearch import selective_search
import os
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from math import atan

#Path: folder that contains the images you want to use
#dirname: folder inside the Path folder that you want to save the new images in
def sel_search(path, dirname, height):

    alphaG = 1737
    alphaA = 1.246
    alphaB = 1.321
    alphaC = 0.732

    screenHeight = 1080
    screenWidth = 1920

    g = alphaG * atan(1.0/height)

    sideLength = int(alphaA * g)

    bigArea = alphaB * g
    smallArea = alphaC * g

    plt.ion()

    listing = os.listdir(path)

    #file = listing[0]
    for file in listing:

        print file

        imHQ = misc.imread(path + file)

        resize = 0.4

        imLowQ = misc.imresize(imHQ, resize, 'cubic')

        img_lbl, regions = selective_search(imLowQ, scale=700, sigma=0.8, min_size=800)

        #img_lbl, regions = selective_search(imLowQ, scale=200, sigma=0.8, min_size=200)

        #img_lbl, regions = selective_search(imLowQ, scale = 50, sigma = 0.7, min_size = 50)

        candidates = set()


        for r in regions:

            newRect = (r['rect'][0]/resize, r['rect'][1]/resize, r['rect'][2]/resize, r['rect'][3]/resize)
            # excluding same rectangle (with different segments)
            if newRect in candidates:
                # print 'Already done'
                continue

            x, y, w, h = newRect

            if w == 0 or h == 0 or w / float(h) > 2 or h / float(w) > 2:
                #print 'Distorted'
                continue

            # excluding regions smaller than smallArea pixels and bigger than bigArea pixels
            #if w * h < smallArea or w * h > bigArea:
            if w * h < 5000 or w * h > 0.9 * screenWidth * screenHeight:
                # print 'Too small'
                continue

            #print 'oldRect:', newRect

            candidates.add(newRect)




        candidates2 = set()

        for c in candidates:

            size = sideLength


            x, y, w, h = c

            print x, y, w, h

            center = (x + w / 2.0, y + h / 2.0)

            leftEdge = max(0, int(center[0] - size/2.0))
            rightEdge = min(screenWidth, int(center[0] + size/2.0))
            topEdge = max(0, int(center[1] - size/2.0))
            bottomEdge = min(screenHeight, int(center[1] + size/2.0))

            cropped = (leftEdge, rightEdge, topEdge, bottomEdge)
            #print 'cropped:', cropped

            squareRect = (cropped[0], cropped[2], cropped [1] - cropped[0], cropped[3] - cropped[2])
            #print 'square: ', squareRect

            candidates2.add(squareRect)

        plt.close('all')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40, 40))
        ax.imshow(imHQ)
        for x, y, w, h in candidates2:
           print x, y, w, h
           rect = mpatches.Rectangle(
               (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
           ax.add_patch(rect)
        plt.pause(0.00000001)
        plt.imshow(imHQ, interpolation='nearest')
        plt.draw()
        plt.show()

        plt.pause(0.2)

        i = 1
        for elem in candidates2:
            #print elem

            cropped_im = imHQ[elem[1]: elem[1] + elem[3], elem[0]: elem[0] + elem[2]]


            file = file.replace('.jpg','')

            #print file

            misc.imsave(path + dirname + '/' + file + str(i) + '.jpg', cropped_im)

            i = i + 1

            print 'image saved!'