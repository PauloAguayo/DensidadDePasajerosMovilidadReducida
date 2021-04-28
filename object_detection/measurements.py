import numpy as np

class Measurements(object):
    def __init__(self,mode):
        self.mode = mode

    def Area(self,polygon_slf,chair_slf,standard_area):
        if self.mode == 1:
            return(float(polygon_slf/chair_slf*standard_area))
        elif self.mode == 2:
            return(float(chair_slf*standard_area/polygon_slf))

    def iou(self,bb_test,bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
          + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        return(o)

    def PolyArea(self,x,y):  # shoelace formula
        return (0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))))

    def Area_Voronoi(self,hull_pol,hull_pol_mini,gt_pol):
        return(float(gt_pol*hull_pol_mini/hull_pol))
