import numpy as np
import cv2 as cv

'''
    place multiple images into one single image template
    :param subimage = (width, height)

'''
class ImageTemp(object):
    def __init__(self, subimage=(60, 60), size=(8, 10), border=5):
        self.subimg_wid = subimage[0]
        self.subimg_height = subimage[1]
        self.height = (self.subimg_height+border)*size[0]+border
        self.width = (self.subimg_wid+border)*size[1]+border
        self.cols = size[1]
        self.rows = size[0]
        self.border = border
        self.img = np.ones((self.height, self.width))

    def show(self, winname):
        cv.imshow(winname, self.img)

    def get_multiImage(self):
        return self.img

    def fill(self, sub_img, place=(0, 0), interpolation=cv.INTER_CUBIC):

        if place[0] > self.rows or place[1] > self.cols:
            print 'location of the subimage exceed the muliImage size'
            return False

        row = place[0]
        col = place[1]
        x = self.border*(col+1) + (col)*self.subimg_wid
        y= self.border*(row+1) + (row)*self.subimg_height

        sub_img = cv.resize(sub_img, (self.subimg_wid, self.subimg_height), interpolation=interpolation)
        self.img[y:y+self.subimg_height, x:x+self.subimg_wid] = sub_img