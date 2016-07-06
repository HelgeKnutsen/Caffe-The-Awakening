# -*- coding: utf-8 -*-

from selSearch import sel_search

baneInn = "/home/mathias/Bilder/testdata"
#baneInn = '/home/ogn/Dropbox/Testbilder 2016-06-29/5m/Med mennesker/'
resultater = "detect"

if baneInn[-1] != "/": #Sjekker om banen slutter med skråstrek
    baneInn += "/"

sel_search(baneInn, resultater, 10)

#this comment