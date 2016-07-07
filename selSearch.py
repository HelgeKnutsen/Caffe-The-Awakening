# -*- coding: utf-8 -*-

from selectivesearch import selective_search
import os
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import math

#Path: folder that contains the images you want to use
#dirname: folder inside the Path folder that you want to save the new images in
def sel_search(path, dirname, height):

    alphaG = 1737
    alphaA = 1.246
    alphaB = 1.321
    alphaC = 0.732

    screenHeight = 1080
    screenWidth = 1920

    g = alphaG * math.atan(1.0/height)

    sideLength = int(alphaA * g)

    bigArea = (alphaB * g)**2
    smallArea = (alphaC * g)**2

    plt.ion()

    listing = os.listdir(path)
    #listing.sort()

    #file = listing[0]

    gammel_tid = time.time()

    for file in listing:

        # Filtere
        filtrer_avlange = True
        filtrer_store = True
        filtrer_naerme = True
        minstoerrelse = True
        vis_bilde = True

        print file

        try:
            imHQ = misc.imread(path + file)
        except IOError: #Dersom filen er en mappe
            print("Fant mappe: " + file)
            continue

        resize = 0.4

        print(g)

        scale = 430.2 * math.e**(0.02050*g*resize) # Funksjon funnet ved regresjon etter forsøk med ulike resize-verdier med 5 m-bilder
        print(scale)

        imLowQ = misc.imresize(imHQ, resize, 'cubic')

        img_lbl, regions = selective_search(imLowQ, scale=scale, sigma=0.8, min_size=150)

        #img_lbl, regions = selective_search(imLowQ, scale=200, sigma=0.8, min_size=200)

        #img_lbl, regions = selective_search(imLowQ, scale = 50, sigma = 0.7, min_size = 50)

        candidates = set()
        sentre = []

        for r in regions:

            newRect = (r['rect'][0]/resize, r['rect'][1]/resize, r['rect'][2]/resize, r['rect'][3]/resize)
            # excluding same rectangle (with different segments)
            if newRect in candidates:
                # print 'Already done'
                continue

            x, y, w, h = newRect

            if w == 0 or h == 0:
                continue

            if filtrer_avlange and (w / float(h) > 2 or h / float(w) > 2):
                #print 'Distorted'
                continue

            # excluding regions smaller than smallArea pixels and bigger than bigArea pixels
            #if w * h < smallArea or w * h > bigArea:
            if filtrer_store and w * h > 0.8 * screenWidth * screenHeight:
                # altfor stor
               continue

            # Ignorer kandidater som er for nærme en annen kandidat
            if filtrer_naerme:
                minsteavstand = 0.4*sideLength
                sentrum = (x + w/2.0, y + h/2.0)
                forNaerme = False
                for annenSentrum in sentre:
                    deltaX = abs(sentrum[0] - annenSentrum[0])
                    deltaY = abs(sentrum[1] - annenSentrum[1])
                    avstand = (deltaX**2 + deltaY**2)**0.5
                    if avstand < minsteavstand:
                        forNaerme = True
                        print("For nærme en annen kandidat.")
                        break
                if forNaerme:
                    continue
                else:
                    sentre.append(sentrum)

            #print 'oldRect:', newRect

            candidates.add(newRect)




        candidates2 = set()

        for c in candidates:

            size = sideLength


            x, y, w, h = c

            print x, y, w, h

            center = (x + w / 2.0, y + h / 2.0)

            #Gjøre om til minstestørrelse
            if minstoerrelse and (w < sideLength):
                leftEdge = max(0, int(center[0] - size/2.0))
                rightEdge = min(screenWidth, int(center[0] + size/2.0))
            else:
                leftEdge = x
                rightEdge = x + w
            if minstoerrelse and (h < sideLength):
                topEdge = max(0, int(center[1] - size/2.0))
                bottomEdge = min(screenHeight, int(center[1] + size/2.0))
            else:
                topEdge = y
                bottomEdge = y + h

            cropped = (leftEdge, rightEdge, topEdge, bottomEdge)
            #print 'cropped:', cropped

            squareRect = (cropped[0], cropped[2], cropped [1] - cropped[0], cropped[3] - cropped[2])
            #print 'square: ', squareRect

            candidates2.add(squareRect)

        if vis_bilde:
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

        ny_tid = time.time()
        print(ny_tid - gammel_tid)
        gammel_tid = ny_tid

        #plt.pause(0.2)

        i = 1
        for elem in candidates2:
            #print elem

            cropped_im = imHQ[elem[1]: elem[1] + elem[3], elem[0]: elem[0] + elem[2]]


            file = file.replace('.jpg','')

            #print file

            try:
                #print(imHQ)
                misc.imsave(path + dirname + '/' + file + "_" + str(i) + '.jpg', cropped_im)
            except IOError: #Dersom mappen ikke fins
                os.mkdir(path + dirname + '/') #Lag mappen

            i = i + 1

            print 'image saved!'