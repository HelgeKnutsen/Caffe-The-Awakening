from selectivesearch import selective_search
import os
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#Path: folder that contains the images you want to use
#dirname: folder inside the Path folder that you want to save the new images in
def sel_search(path, dirname):


    listing = os.listdir(path)

    #file = listing[0]
    for file in listing:

        print file

        image = misc.imread(path + file)

        img_lbl, regions = selective_search(image, scale=1000, sigma=0.9, min_size=1000)

        candidates = set()


        for r in regions:
            #print r['size']
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                # print 'Already done'
                continue

            # distorted rects
            x, y, w, h = r['rect']
            
            if w == 0 or h == 0 or w / float(h) > 2 or h / float(w) > 2:
                #print 'Distorted'
                continue

            # excluding regions smaller than 2000 pixels
            if w * h < 3600 or w * h > 28000:
            #if w * h < 100:
                # print 'Too small'
                continue

            candidates.add(r['rect'])

        #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40, 40))
        #ax.imshow(image)
        #for x, y, w, h in candidates:
        #    print x, y, w, h
        #    rect = mpatches.Rectangle(
        #        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        #    ax.add_patch(rect)

        #plt.imshow(image, interpolation='nearest')
        #plt.draw()
        #plt.show()

        i = 1
        for elem in candidates:
            print elem
            cropped_im = image[elem[1]: elem[1] + elem[3], elem[0]: elem[0] + elem[2]]

            file = file.replace('.jpg','')

            print file

            misc.imsave(path + dirname + '/' + file + str(i) + '.jpg', cropped_im)

            i = i + 1

            print 'image saved!'