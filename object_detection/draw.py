import cv2
from measurements import Measurements
import numpy as np
from object_detection.utils import visualization_utils as vis_util
from shapely import geometry
from scipy.spatial import Voronoi, ConvexHull
import pandas as pd


class Drawing(object):
    def __init__(self,mode,image,copy_image,final_image,min_score,angle):
        self.image = image
        self.copy_image = copy_image
        self.final_image = final_image
        self.image_center = np.array([int(self.image.shape[0]/2),int(self.image.shape[1]/2)]) # (y,x)
        self.image_bottom = np.array([int(self.image.shape[0]),int(self.image.shape[1]/2)]) # (y,x)
        self.min_score = min_score
        self.min_length = np.minimum(self.image.shape[0],self.image.shape[1])
        self.max_length = np.maximum(self.image.shape[0],self.image.shape[1])
        self.mode = mode
        self.measures = Measurements(self.mode)
        self.copy = self.image.copy()
        self.origin = np.array([0,0])
        self.angle = angle

    def Prepare_data(self,scores,boxes,classes):
        self.scores = scores
        self.boxes = boxes
        self.classes = classes

    def Draw_detections(self,input1,n,H,h):
        print("------ DRAWING DETECTIONS AND OPTIMIZING PROJECTIONS ------")
        self.rec_points = []
        def Quadrant(y,x,beta,slope,photo,d,caja,clase): # (y,x) centroid detection box

            # normalized coordinates
            x_param = 2*x/self.image.shape[1]-1
            y_param = 2*y/self.image.shape[0]-1
            y_axis = []

            if x_param>=0 and y_param>=0: # 4th
                if d==0:
                    x_axis = np.arange(self.image_bottom[1],x+1)
                    if slope >1: slope = abs(slope) - abs(int(slope))
                else:
                    x_axis = np.arange(self.image_bottom[0],y+1)
                    if slope <1: slope+=1
                x_axis = x_axis[::-1]
            elif x_param<0 and y_param>=0:  # 3rd
                if d==0:
                    x_axis = np.arange(x,self.image_bottom[1]+1)                            #    2  |  1
                    if abs(slope) >1: slope = (-1)*(abs(slope) - abs(int(slope)))           #  _____|_____
                else:                                                                       #    3  |  4      quadrants
                    x_axis = np.arange(self.image_bottom[0],y+1)                            #       |
                    x_axis = x_axis[::-1]
                    if abs(slope)<1: slope-=1
            elif x_param>=0 and y_param<0:   # 1st
                if d==0:
                    x_axis = np.arange(self.image_bottom[1],x+1)
                    if abs(slope) >1: slope = (-1)*(abs(slope) - abs(int(slope)))
                    x_axis = x_axis[::-1]
                else:
                    x_axis = np.arange(y,self.image_bottom[0]+1)
                    if abs(slope) <1: slope-=1
            elif x_param<0 and y_param<0:    # 2nd
                if d==0:
                    x_axis = np.arange(x,self.image_bottom[1]+1)
                    if slope >1: slope = slope - int(slope)
                else:
                    x_axis = np.arange(y,self.image_bottom[0]+1)
                    if slope <1: slope+=1

            if d == 0:
                def straight(x__,x,y,slope):
                    return(slope*(x__-x)+y)
            else:
                def straight(x__,x,y,slope): # do not get confused!!! this x__ is a y__
                    return((x__-y)/float(slope)+x)

            for x__ in x_axis:
                y__ = straight(x__,self.image_bottom[1],self.image_bottom[0],slope)
                y_axis.append(y__)
            y_axis = np.array(y_axis)

            # optimization points
            lim = 0
            beta*=(self.max_length/self.angle)
            coords = [-100,-100]
            for x_p,y_p in zip(x_axis,y_axis):
                if d == 0:
                    bottom2Coord = np.linalg.norm(np.array([y_p,x_p])-self.image_bottom)
                    #cv2.circle(photo,(int(x_p),int(y_p)),1,(255,255,0),-1)
                else:
                    bottom2Coord = np.linalg.norm(np.array([x_p,y_p])-self.image_bottom)
                    #cv2.circle(photo,(int(y_p),int(x_p)),1,(255,255,0),-1)
                if bottom2Coord<=abs(beta):
                    if bottom2Coord>lim:
                        lim = bottom2Coord
                        if d == 0:
                            coords = [y_p,x_p]
                        else:
                            coords = [x_p,y_p]
            # coords y,x
            #cv2.circle(photo,(int(coords[1]),int(coords[0])),3,(255,0,0),-1)
            geom_coords = geometry.Point([coords[1],coords[0]])
            if self.poly.contains(geom_coords):
                self.rec_points.append([int(coords[0]),int(coords[1])])
                vis_util.draw_bounding_box_on_image_array(photo,caja[0],caja[1],caja[2],caja[3],color='red',thickness=4,display_str_list=()) # HEADS
                cv2.circle(photo,(int(coords[1]),int(coords[0])),3,(255,0,0),-1)
                #cv2.putText(photo, str(self.zones), (int(coords[1]-10),int(coords[0]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                return(1)
            return(0)

        people = 0
        if n==1:
            photo = self.image
        elif n==0:
            photo = self.copy_image
        else:
            photo = self.final_image

        #cv2.circle(photo,(self.image_center[1],self.image_center[0]),7,(255,255,255),-1)
        for (puntaje,caja,clase) in zip(self.scores,self.boxes,self.classes):
            distance_x = caja[3] - caja[1]
            distance_y = caja[2] - caja[0]
            point = np.array([(caja[1]+distance_x/2.0)*photo.shape[1],(caja[0]+distance_y/2.0)*photo.shape[0]]) # (x,y)   centroid box
            if point[1]<0: point[1] = 0
            if point[0]<0: point[0] = 0

            if (puntaje>=self.min_score) and (clase==1.0 or clase==4.0):
                # pixel's distance to radians
                gamma = np.linalg.norm(np.array([point[1],point[0]])-self.image_bottom)
                gamma *= (self.angle/self.max_length)

                d1_prima = np.tan(gamma)*H
                d1 = d1_prima - np.tan(gamma)*h
                alpha = np.arctan(d1/H)
                beta = abs(gamma) - abs(alpha)
                slope = (point[1]-self.image_bottom[0])/(point[0]-self.image_bottom[1])
                if abs(slope)>=1:
                    d = 1
                elif abs(slope)<1:
                    d = 0
                people+=Quadrant(int(point[1]),int(point[0]),float(beta),slope,photo,d,caja,clase)
            elif clase==2.0:
                x = np.array([caja[1]+distance_x*input1/2.0,caja[1]+distance_x*(1-input1/2.0),caja[1]+distance_x*(1-input1/2.0),caja[1]+distance_x*input1/2.0])*photo.shape[1]
                y = np.array([caja[0]+distance_y*input1/2.0,caja[0]+distance_y*input1/2.0,caja[0]+distance_y*(1-input1/2.0),caja[0]+distance_y*(1-input1/2.0)])*photo.shape[0]
                vis_util.draw_bounding_box_on_image_array(photo,caja[0]+distance_y*input1/2.0,caja[1]+distance_x*input1/2.0,caja[2]-distance_y*input1/2.0,caja[3]-distance_x*input1/2.0,color='yellow',thickness=4,display_str_list=()) # WHEELCHAIRS
                wheelchair_area = self.measures.PolyArea(x,y)
        return([wheelchair_area,people])

    def Generate_Polygon(self, nameWindow):
        def draw_circle(event,x,y,flags,param):
            global mouseX,mouseY
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(self.copy_image,(x,y),6,(0,255,255),-1)
                mouseX,mouseY = x,y

        cv2.namedWindow(nameWindow)
        cv2.setMouseCallback(nameWindow,draw_circle)
        points = []
        point_counter = 0
        while(1):
            cv2.imshow(nameWindow,self.copy_image)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('a'):
                point_counter+=1
                points.append([mouseX,mouseY])
                cv2.circle(self.copy_image,(mouseX,mouseY),8,(0,0,0),3)
                cv2.putText(self.copy_image, str(point_counter), (mouseX,mouseY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            elif k == ord('r'):
                for i,c in zip(points,np.arange(1,point_counter+1)):
                    cv2.circle(self.copy_image,(i[0],i[1]),8,(255,255,255),-1)
                    cv2.line(self.copy_image,(i[0]-6,i[1]-6),(i[0]+6,i[1]+6),(0,0,0),2)
                    cv2.line(self.copy_image,(i[0]+6,i[1]-6),(i[0]-6,i[1]+6),(0,0,0),2)
                    cv2.putText(self.copy_image, str(c), (i[0],i[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                points = []
                point_counter = 0
        cv2.destroyAllWindows()

        self.poly = geometry.Polygon(points)
        self.centroid = np.array(list(self.poly.centroid.coords)[0]) # x,y

        extended_points = []
        for punto in points:
            if (punto[0] <= self.centroid[0]) and (punto[1] <= self.centroid[1]):
                extended_points.append([punto[0]-1,punto[1]-1])
            elif (punto[0] <= self.centroid[0]) and (punto[1] > self.centroid[1]):
                extended_points.append([punto[0]-1,punto[1]+1])
            elif (punto[0] > self.centroid[0]) and (punto[1] <= self.centroid[1]):
                extended_points.append([punto[0]+1,punto[1]-1])
            elif (punto[0] > self.centroid[0]) and (punto[1] > self.centroid[1]):
                extended_points.append([punto[0]+1,punto[1]+1])

        self.poly_extended = geometry.Polygon(extended_points)

        self.image_bottom[1] = self.centroid[0]

        self.poly_points = []
        for c,(mx,my) in enumerate(points):
            if c!=0:
                mx_prev = points[c-1][0]
                my_prev = points[c-1][1]

                if my>=my_prev:
                    y_axis = np.arange(my_prev,my+1)
                else:
                    y_axis = np.arange(my,my_prev+1)
                    y_axis = y_axis[::-1]

                if mx>=mx_prev:
                    x_axis = np.arange(mx_prev,mx+1)
                else:
                    x_axis = np.arange(mx,mx_prev+1)
                    x_axis = x_axis[::-1]

                slope = (my-my_prev)/(mx-mx_prev)

                if len(x_axis) >= len(y_axis):
                    y_axis = []
                    for x in x_axis:
                        self.poly_points.append([round(x),round((-1)*(slope*(mx-x)-my))])
                else:
                    x_axis = []
                    for y in y_axis:
                        self.poly_points.append([round((-1)*((my-y)/slope-mx)),round(y)])

            if c==len(points)-1:
                mx_post = points[0][0]
                my_post = points[0][1]

                if my_post>=my:
                    y_axis = np.arange(my,my_post+1)
                else:
                    y_axis = np.arange(my_post,my+1)
                    y_axis = y_axis[::-1]

                if mx_post>=mx:
                    x_axis = np.arange(mx,mx_post+1)
                else:
                    x_axis = np.arange(mx_post,mx+1)
                    x_axis = x_axis[::-1]

                slope = (my_post-my)/(mx_post-mx)

                if len(x_axis) >= len(y_axis):
                    y_axis = []
                    for x in x_axis:
                        self.poly_points.append([round(x),round((-1)*(slope*(mx_post-x)-my_post))])
                else:
                    x_axis = []
                    for y in y_axis:
                        self.poly_points.append([round((-1)*((my_post-y)/slope-mx_post)),round(y)])

        self.poly_points = self.poly_points[:-1]
        return(points)

    def Handcrafted(self,dec):
        self.drawing = False # True if mouse is pressed
        self.ix,self.iy = -1,-1

        def draw_square(event,x,y,flags,param):
            global xx,yy

            if x<0: x = 0
            if y<0: y = 0

            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.ix,self.iy = x,y
                cv2.line(self.copy_image,(self.ix-30,self.iy),(self.ix+30,self.iy),(0,0,0),2)
                cv2.line(self.copy_image,(self.ix,self.iy-30),(self.ix,self.iy+30),(0,0,0),2)
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing == True:
                    dif_x = x-self.ix
                    dif_y = y-self.iy
                    cv2.line(self.copy_image,(self.ix,self.iy),(self.ix+dif_x,self.iy),(0,0,0),2)
                    cv2.line(self.copy_image,(self.ix,self.iy),(self.ix,self.iy+dif_y),(0,0,0),2)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                xx,yy = x,y
                end_point = np.array([xx,yy])
                if np.linalg.norm(end_point-self.origin)<np.linalg.norm([self.ix,self.iy]-self.origin):
                    cv2.rectangle(self.copy_image,(x,y),(self.ix,self.iy),(0,255,0),3)
                elif np.linalg.norm(end_point-self.origin)>np.linalg.norm([self.ix,self.iy]-self.origin):
                    cv2.rectangle(self.copy_image,(self.ix,self.iy),(x,y),(0,255,0),3)
        bx = []
        clss = []
        scr = []
        while(dec=='y'):
            cv2.namedWindow('Handcrafted image')
            cv2.setMouseCallback('Handcrafted image',draw_square)
            cv2.imshow('Handcrafted image',self.copy_image)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord('a'): # Add heads
                text = "Added head"
                cv2.rectangle(self.copy_image,(self.ix,self.iy),(xx,yy),(0,0,255),3)
                cv2.putText(self.copy_image, text, (self.ix , self.iy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                centr = np.array([int(abs(self.ix-xx)),int(abs(self.iy-yy))])
                xxx = np.minimum(xx,self.ix)
                yyy = np.minimum(yy,self.iy)
                bx.append([yyy/self.copy_image.shape[0],xxx/self.copy_image.shape[1],(yyy+centr[1])/self.copy_image.shape[0],(xxx+centr[0])/self.copy_image.shape[1]])
                clss.append([4.0])
                scr.append([0.99])
            elif k == ord('q'): # Add wheelchairs
                text = "Added wheelchair"
                cv2.rectangle(self.copy_image,(self.ix,self.iy),(xx,yy),(0,255,0),3)
                cv2.putText(self.copy_image, text, (self.ix , self.iy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                self.classes[self.classes==2.0]=0.0
                bx.append([self.iy/self.copy_image.shape[0],self.ix/self.copy_image.shape[1],yy/self.copy_image.shape[0],xx/self.copy_image.shape[1]])
                clss.append([2.0])
                scr.append([0.99])
            elif k == ord('z'): # Remove heads
                for c,x in enumerate(self.boxes):
                    bb_gt = [x[1],x[0],x[3],x[2]]
                    bb_remove = [self.ix/self.copy_image.shape[1],self.iy/self.copy_image.shape[0],xx/self.copy_image.shape[1],yy/self.copy_image.shape[0]]
                    if self.measures.iou(bb_remove,bb_gt)>0.5:
                        text = "Detection removed"
                        cv2.rectangle(self.copy_image,(self.ix,self.iy),(xx,yy),(255,0,255),3)
                        cv2.putText(self.copy_image, text, (self.ix , self.iy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
                        self.classes[c]=3.0
        cv2.destroyAllWindows()
        try:
            bx = np.array(bx)
            self.boxes = np.concatenate((self.boxes,bx))
            clss = np.array(clss)
            self.classes = np.concatenate((self.classes,clss))
            scr = np.array(scr)
            self.scores = np.concatenate((self.scores,scr))
        except:
            self.scores = self.scores
            self.classes = self.classes
            self.boxes = self.boxes

    def Voronoi_diagram(self,image):
        rec_points = np.array(self.rec_points)   # proyecciones
        rec_points = np.flip(rec_points)
        vor = Voronoi(rec_points)
        polygon_number = 0

        finite_ridges = []
        finite_points = []
        for simplex in vor.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):

                finite_ridges.append(simplex)
                x_s = vor.vertices[simplex, 0]
                y_s = vor.vertices[simplex, 1]

                if y_s[1]>=y_s[0]:
                    y_axis = np.arange(y_s[0],y_s[1]+1)
                else:
                    y_axis = np.arange(y_s[1],y_s[0]+1)
                    y_axis = y_axis[::-1]

                if x_s[1]>=x_s[0]:
                    x_axis = np.arange(x_s[0],x_s[1]+1)
                else:
                    x_axis = np.arange(x_s[1],x_s[0]+1)
                    x_axis = x_axis[::-1]

                slope = (y_s[1]-y_s[0])/(x_s[1]-x_s[0])

                if len(x_axis) >= len(y_axis):
                    y_axis = []
                    for x in x_axis:
                        y_axis.append((-1)*(slope*(x_s[1]-x)-y_s[1]))
                else:
                    x_axis = []
                    for y in y_axis:
                        x_axis.append((-1)*((y_s[1]-y)/slope-x_s[1]))

                fin = []
                for y,x in np.transpose(np.array([y_axis,x_axis])):
                    pnt = geometry.Point([round(x),round(y)])
                    if self.poly_extended.contains(pnt):
                        cv2.circle(image,(round(x),round(y)),1,(0,255,255),-1)
                        fin.append([round(x),round(y)])
                finite_points.append(fin)

        center = rec_points.mean(axis=0)
        infinite_ridges = []
        infinite_points = []
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.any(simplex < 0):
                infinite_ridges.append(simplex)
                i = simplex[simplex >= 0][0] # finite end Voronoi vertex
                t = rec_points[pointidx[1]] - rec_points[pointidx[0]]  # tangent
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]]) # normal
                midpoint = rec_points[pointidx].mean(axis=0)
                p_f = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 1000
                p_i = np.array([round(vor.vertices[i,0]),round(vor.vertices[i,1])])  # x,y     # vertices infinitos de voronoi

                if p_f[1]>=p_i[1]:
                    y_axis = np.arange(p_i[1],p_f[1]+1)
                else:
                    y_axis = np.arange(p_f[1],p_i[1]+1)
                    y_axis = y_axis[::-1]
                if p_f[0]>=p_i[0]:
                    x_axis = np.arange(p_i[0],p_f[0]+1)
                else:
                    x_axis = np.arange(p_f[0],p_i[0]+1)
                    x_axis = x_axis[::-1]

                zlope = (p_f[1]-p_i[1])/(p_f[0]-p_i[0])

                if len(x_axis) >= len(y_axis):
                    y_axis = []
                    for x in x_axis:
                        y_axis.append((-1)*(zlope*(p_f[0]-x)-p_f[1]))
                else:
                    x_axis = []
                    for y in y_axis:
                        x_axis.append((-1)*((p_f[1]-y)/zlope-p_f[0]))

                inf = []
                for y,x in np.transpose(np.array([y_axis,x_axis])):
                    pnt = geometry.Point([round(x),round(y)])
                    if self.poly_extended.contains(pnt):
                        cv2.circle(image,(round(x),round(y)),1,(0,255,255),-1)
                        inf.append([round(x),round(y)])
                infinite_points.append(inf)

        cv2.imshow('Area selection',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        regions = vor.regions
        polygons = []
        polygon_area = 0
        for r in regions:
            r = np.array(r)
            pp = []
            if len(r)!=0:
                pp = []
                if np.all(r >= 0):
                    for c,(ridge_1,ridge_2) in enumerate(finite_ridges):
                        if (ridge_1 in r) and (ridge_2 in r):
                            for kj in finite_points[c]:
                               pp.append(kj)
                else:
                    for c,(ridge_1,ridge_2) in enumerate(infinite_ridges):

                        if ridge_1<0: ridgee = ridge_2
                        else:   ridgee = ridge_1

                        if ridgee in r:
                            for kj in infinite_points[c]:
                                pp.append(kj)
                                cv2.circle(image,(round(kj[0]),round(kj[1])),1,(0,255,255),-1)

                    for c,(ridge_1,ridge_2) in enumerate(finite_ridges):
                        if (ridge_1 in r) and (ridge_2 in r):
                            for kj in finite_points[c]:
                                pp.append(kj)
                                cv2.circle(image,(round(kj[0]),round(kj[1])),1,(0,255,255),-1)

                    def str_to(pp):
                        return([str(kk) for kk in pp])

                    arreglo_c_correct = []
                    def_points = []
                    for c,kn in enumerate(self.poly_points):
                        kn = str(kn)
                        if kn in str_to(pp):
                            arreglo_c_correct.append(c)

                    if len(arreglo_c_correct)>1:

                        for ff,kkk in enumerate(arreglo_c_correct):
                            if ff!=0:
                                for i in self.poly_points[arreglo_c_correct[ff-1]:arreglo_c_correct[ff]+1]:
                                    def_points.append(i)

                        litsa = []
                        for kg in self.poly_points[arreglo_c_correct[1]:]:
                            litsa.append(kg)
                        for kg in self.poly_points[:arreglo_c_correct[0]+1]:
                            litsa.append(kg)

                        if len(litsa)>len(def_points):
                            for o in def_points:
                                pp.append(o)
                                cv2.circle(image,(round(o[0]),round(o[1])),1,(0,255,255),-1)
                        else:
                            for o in litsa:
                                pp.append(o)
                                cv2.circle(image,(round(o[0]),round(o[1])),1,(0,255,255),-1)

                hull = ConvexHull(pp)
                for simplex in hull.simplices:
                    p_x_s = pp[simplex[0]][0]
                    p_y_s = pp[simplex[0]][1]
                    p_x_f = pp[simplex[1]][0]
                    p_y_f = pp[simplex[1]][1]

                    #cv2.line(image,(p_x_s,p_y_s),(p_x_f,p_y_f),(255,0,0),2)
                cx = np.mean(hull.points[hull.vertices,0])
                cy = np.mean(hull.points[hull.vertices,1])
                polygons.append(hull.volume)
                polygon_area += hull.volume
                polygon_number+=1
                cv2.putText(image, str(polygon_number), (round(cx-10),round(cy+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        for c,p in enumerate(polygons):
            print('Polygon '+str(c+1)+ ' Area = ' + str(self.measures.Area_Voronoi(polygon_area,p,)))

        cv2.imshow('Area selection',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return(polygon_area)
