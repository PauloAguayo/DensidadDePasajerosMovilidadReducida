import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from shapely import geometry
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", required=True, help="path to object detection model")
parser.add_argument("-l", "--labels", required=True, help="path to labels file")
parser.add_argument("-i", "--input", default=0, type=str, help="path to optional input image file", required=True)
parser.add_argument("-o", "--output", type=str, help="path to optional output image file")
parser.add_argument("-c", "--threshold", type=float, default=0.8, help="minimum probability to filter weak detection")
parser.add_argument("-d", "--distance", type=float, default=0.7, help="segmentation percentage for distance criteria")
parser.add_argument("-wr", "--wheelchair_rear", type=float, default=0.8, help="size of wheelchair rear in meters")
parser.add_argument("-ws", "--wheelchair_side", type=float, default=1.2, help="size of wheelchair side in meters")
parser.add_argument("-mode", "--mode", type=int, required=True, help="selection mode: 1 = wheelchair ground truth; 2 = polygon ground truth")
args = parser.parse_args()
vargs = vars(args)

# Selecting the mode program
if vargs["mode"] == 1: # the wheelchair_rear and wheelchair_side are needed
    wheelchair_rear = vargs["wheelchair_rear"]
    wheelchair_side = vargs["wheelchair_side"]
    standard_area = wheelchair_side*wheelchair_rear
    area_name = "POLYGON AREA CALCULATED ="
    print('-------------------------------------------------------------------------')
    print('----- WHEELCHAIR AREA AS GROUND TRUTH MEASURE -----')
    print('-------------------------------------------------------------------------')
    print('WHEELCHAIR AREA =', standard_area, 'm2')
    print('-------------------------------------------------------------------------')
elif vargs["mode"] == 2:
    area_name = "WHEELCHAIR AREA CALCULATED ="
    print('-------------------------------------------------------------------------')
    print('----- POLYGON AREA AS GROUND TRUTH MEASURE -----')
    print('-------------------------------------------------------------------------')

# Function for specific area calculation regarding the mode selection
def area(standard_area,polygon_slf,chair_slf,mode):
    if mode == 1:
        return(float(polygon_slf/chair_slf*standard_area))
    elif mode == 2:
        return(float(chair_slf/polygon_slf*standard_area))

# Intersection over union
def iou(bb_test,bb_gt): # Computes IUO between two bboxes in the form [x1,y1,x2,y2]
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

# Defining shoelace formula for measuring the selected polygon area
def PolyArea(x,y):  # SHOELACE FORMULA
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# Function for drawing a circle in polygon selection area
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(copy_image,(x,y),6,(0,255,255),-1)
        mouseX,mouseY = x,y

# Function to make new square selections, for manual detections
drawing = False # True if mouse is pressed
ix,iy = -1,-1
# Mouse callback function
def draw_square(event,x,y,flags,param):
    global ix,iy,xx,yy,drawing
    if x<0:
        x = 0
    elif y<0:
        y = 0
    elif x<0 and y<0:
        x,y = 0,0
    #################
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(image,(ix,iy),(x,y),(0,255,0),5)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        xx,yy = x,y
        cv2.rectangle(image,(ix,iy),(x,y),(0,255,0),5)

# Defining path and name of output picture
if vargs["output"]==None:
    output_variable = 'output.jpg'
else:
    output_variable = vargs["output"]

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Storing paths
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,vargs["model"])
PATH_TO_LABELS = os.path.join(CWD_PATH,vargs["labels"])
PATH_TO_IMAGE = os.path.join(CWD_PATH,vargs["input"])

# Number of classes in the trained model
NUM_CLASSES = 2

# Loading the label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Loading Tensorflow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

# Defining input and output tensors for detections
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Loading image
# Expanding image dimensions to have shape: [1, None, None, 3]
image = cv2.imread(PATH_TO_IMAGE)
copy_image = image.copy()
def_image = image.copy()
correction_image = image.copy()
recorrection_image = image.copy()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# Generate polygon
cv2.namedWindow('Image '+vargs["input"])
cv2.setMouseCallback('Image '+vargs["input"],draw_circle)
points = []
while(1):
    cv2.imshow('Image '+vargs["input"],copy_image)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        points.append([mouseX,mouseY])
pts = np.array(points, np.int32)
poly = geometry.Polygon(pts)
centroid = np.array(list(poly.centroid.coords)[0])
if vargs["mode"] == 2:
    print('-------------------------------------------------------------------------')
    standard_area = float(input('MEASURE OF THE POLYGON AREA? (GROUND TRUTH) ' ))
cv2.destroyAllWindows()

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Reshaping useful arrays about details of detections
boxes = boxes.reshape(boxes.shape[1],4)
scores = scores.reshape(scores.shape[1],1)
classes = classes.reshape(classes.shape[1],1)

# Ensuring there is only one wheelchair in detections, the one with the highest score
check_classes = np.argwhere(classes==2.0)
check_details = []
for i,j in check_classes:
    check_details.append([scores[i,j],[i,j]])
check_details = np.array(check_details)
max_class2 = np.amax(check_details, axis=0)
loc = np.argwhere(check_details==max_class2[0])
loc_top2 = check_details[loc[0][0],loc[0][1]+1]
classes[classes==2.0]=0.0
classes[loc_top2[0]]=2.0

# Performing the detections and drawing them into the image
min_score = vargs["threshold"]#0.6
people = 0
min_length = np.minimum(image.shape[0],image.shape[1])
for puntaje,caja,clase in zip(scores,boxes,classes):
    distance_x = caja[3] - caja[1]
    distance_y = caja[2] - caja[0]
    point = np.array([(caja[1]+distance_x)*image.shape[1],(caja[0]+distance_y)*image.shape[0]])
    dist = np.linalg.norm(centroid-point)
    if (puntaje>min_score) and (clase==1.0) and (dist<=min_length*vargs["distance"]):
        people+=1
        vis_util.draw_bounding_box_on_image_array(copy_image,caja[0],caja[1],caja[2],caja[3],color='red',thickness=4,display_str_list=()) # HEADS
    elif clase==2.0:
        x = np.array([caja[1],caja[1]+distance_x,caja[1]+distance_x,caja[1]])*image.shape[1]
        y = np.array([caja[0],caja[0],caja[0]+distance_y,caja[0]+distance_y])*image.shape[0]
        vis_util.draw_bounding_box_on_image_array(copy_image,caja[0],caja[1],caja[2],caja[3],color='yellow',thickness=4,display_str_list=()) # WHEELCHAIRS
        wheelchair_area = PolyArea(x,y)
print('-------------------------------------------------------------------------')
print('PEOPLE DETECTED =',people)
print('-------------------------------------------------------------------------')

# Drawing polygon and its centroid
cv2.circle(copy_image,(int(centroid[0]),int(centroid[1])),6,(0,255,255),-1)
polygon_area = PolyArea(pts[:,0],pts[:,1])
pts = pts.reshape((-1,1,2))
cv2.polylines(copy_image,[pts],True,(0,255,255))
cv2.imshow('Area selection',copy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In case detections are incomplete, you can make square selections to ensure a better performance
input1 = input('HANDCRAFTED SELECTIONS? ' )
print('-------------------------------------------------------------------------')
cv2.namedWindow('Handcrafted image')
cv2.setMouseCallback('Handcrafted image',draw_square)
agregate = 0
while(input1=='y'):
    cv2.imshow('Handcrafted image',image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('a'): # Add heads
        boxes = np.concatenate((boxes,np.array([[iy/image.shape[0],ix/image.shape[1],yy/image.shape[0],xx/image.shape[1]]])))
        scores = np.concatenate((scores,np.array([[0.99]])))
        classes = np.concatenate((classes,np.array([[1.0]])))
        agregate-=1
    elif k == ord('q'): # Add wheelchairs
        classes[classes==2.0]=0.0
        boxes = np.concatenate((boxes,np.array([[iy/image.shape[0],ix/image.shape[1],yy/image.shape[0],xx/image.shape[1]]])))
        scores = np.concatenate((scores,np.array([[0.99]])))
        classes = np.concatenate((classes,np.array([[2.0]])))
    elif k == ord('z'): # Remove heads
        for c,x in enumerate(boxes):
            bb_gt = [x[1],x[0],x[3],x[2]]
            bb_remove = [ix/image.shape[1],iy/image.shape[0],xx/image.shape[1],yy/image.shape[0]]
            if iou(bb_remove,bb_gt)>0.5:
                classes[c]=3.0
cv2.destroyAllWindows()

# Drawing old detections plus new handcrafted detections
people = 0
for puntaje,caja,clase in zip(scores,boxes,classes):
    distance_x = caja[3] - caja[1]
    distance_y = caja[2] - caja[0]
    point = np.array([(caja[1]+distance_x)*image.shape[1],(caja[0]+distance_y)*image.shape[0]])
    dist = np.linalg.norm(centroid-point)
    if (puntaje>min_score) and (clase==1.0) and (dist<=min_length*vargs["distance"]):
        people+=1
        vis_util.draw_bounding_box_on_image_array(def_image,caja[0],caja[1],caja[2],caja[3],color='red',thickness=4,display_str_list=()) # HEADS
    elif clase==2.0:
        x = np.array([caja[1],caja[1]+distance_x,caja[1]+distance_x,caja[1]])*image.shape[1]
        y = np.array([caja[0],caja[0],caja[0]+distance_y,caja[0]+distance_y])*image.shape[0]
        vis_util.draw_bounding_box_on_image_array(def_image,caja[0],caja[1],caja[2],caja[3],color='yellow',thickness=4,display_str_list=()) # WHEELCHAIRS
        wheelchair_area = PolyArea(x,y)

# This boxes are added separately because maybe this are detections whom don't accomplish with the "dist" parameter
if agregate != 0:
    for caja in boxes[agregate:]:
        people+=1
        vis_util.draw_bounding_box_on_image_array(def_image,caja[0],caja[1],caja[2],caja[3],color='red',thickness=4,display_str_list=())
cv2.imshow('Image',def_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculation outcomes
total_area = area(standard_area,polygon_area,wheelchair_area,vargs["mode"])
if vargs["mode"] == 1:
    people_density = people/total_area
elif vargs["mode"] == 2:
    people_density = people/standard_area
print('PEOPLE DETECTED =',people)
print(area_name, total_area, 'm2')
print('DENSITY = ', people_density, 'people/m2')
print('------------------------------------------------------------------------')

# Loop for bounding box correction for wheelchair detection following the same last procedure for plotting squares detections
iter = 1
input2 = input('WHEELCHAIR AREA CORRECTION? ') # Percentage input
print('-------------------------------------------------------------------------')
dec = True
while(input2=='y'):
    input1 = float(input('CORRECTION VALUE? (PERCENTAGE) '))
    print('-------------------------------------------------------------------------')
    for puntaje,caja,clase in zip(scores,boxes,classes):
        distance_x = (caja[3] - caja[1])
        distance_y = caja[2] - caja[0]
        point = np.array([(caja[1]+distance_x)*image.shape[1],(caja[0]+distance_y)*image.shape[0]])
        dist = np.linalg.norm(centroid-point)
        if (puntaje>min_score) and (clase==1.0) and (dist<=min_length*vargs["distance"]):
            vis_util.draw_bounding_box_on_image_array(correction_image,caja[0],caja[1],caja[2],caja[3],color='red',thickness=4,display_str_list=()) # HEADS
        elif clase==2.0:
            x = np.array([caja[1]+distance_x*input1/2.0,caja[1]+distance_x*(1-input1/2.0),caja[1]+distance_x*(1-input1/2.0),caja[1]+distance_x*input1/2.0])*image.shape[1]        ##
            y = np.array([caja[0]+distance_y*input1/2.0,caja[0]+distance_y*input1/2.0,caja[0]+distance_y*(1-input1/2.0),caja[0]+distance_y*(1-input1/2.0)])*image.shape[0]                    ##
            vis_util.draw_bounding_box_on_image_array(correction_image,caja[0]+distance_y*input1/2.0,caja[1]+distance_x*input1/2.0,caja[2]-distance_y*input1/2.0,caja[3]-distance_x*input1/2.0,color='yellow',thickness=4,display_str_list=()) # WHEELCHAIRS
            wheelchair_area = PolyArea(x,y)
    if agregate != 0:
        for caja,clase in zip(boxes[agregate:],classes[agregate:]):
            if clase==1.0:
                vis_util.draw_bounding_box_on_image_array(correction_image,caja[0],caja[1],caja[2],caja[3],color='red',thickness=4,display_str_list=())
    cv2.imshow('Correction n_'+str(iter),correction_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    total_area = area(standard_area,polygon_area,wheelchair_area,vargs["mode"])
    if vargs["mode"] == 1:
        people_density = people/total_area
    elif vargs["mode"] == 2:
        people_density = people/standard_area
    print('PEOPLE DETECTED =',people)
    print(area_name, total_area, 'm2')
    print('DENSITY = ', people_density, 'people/m2' )
    print('--------------------------------------------------------------------')
    iter+=1
    dec = False
    input2 = input('DO YOU NEED A NEW CORRECTION? ')
    print('-------------------------------------------------------------------------')

# Getting the final picture to save it as the output_variable for practical and analytical uses
if dec == True: input1 = 0
for puntaje,caja,clase in zip(scores,boxes,classes):
    distance_x = (caja[3] - caja[1])
    distance_y = caja[2] - caja[0]
    point = np.array([(caja[1]+distance_x)*image.shape[1],(caja[0]+distance_y)*image.shape[0]])
    dist = np.linalg.norm(centroid-point)
    if (puntaje>min_score) and (clase==1.0) and (dist<=min_length*vargs["distance"]):
        vis_util.draw_bounding_box_on_image_array(recorrection_image,caja[0],caja[1],caja[2],caja[3],color='red',thickness=4,display_str_list=()) # HEADS
    elif (puntaje>min_score) and (clase==2.0):
        x = np.array([caja[1]+distance_x*input1/2.0,caja[1]+distance_x*(1-input1/2.0),caja[1]+distance_x*(1-input1/2.0),caja[1]+distance_x*input1/2.0])*image.shape[1]        ##
        y = np.array([caja[0]+distance_y*input1/2.0,caja[0]+distance_y*input1/2.0,caja[0]+distance_y*(1-input1/2.0),caja[0]+distance_y*(1-input1/2.0)])*image.shape[0]                    ##
        vis_util.draw_bounding_box_on_image_array(recorrection_image,caja[0]+distance_y*input1/2.0,caja[1]+distance_x*input1/2.0,caja[2]-distance_y*input1/2.0,caja[3]-distance_x*input1/2.0,color='yellow',thickness=4,display_str_list=()) # WHEELCHAIRS
        wheelchair_area = PolyArea(x,y)
if agregate != 0:
    for caja,clase in zip(boxes[agregate:],classes[agregate:]):
        if clase==1.0:
            vis_util.draw_bounding_box_on_image_array(recorrection_image,caja[0],caja[1],caja[2],caja[3],color='red',thickness=4,display_str_list=())
cv2.circle(recorrection_image,(int(centroid[0]),int(centroid[1])),6,(0,255,255),-1)
cv2.polylines(recorrection_image,[pts],True,(0,255,255))
cv2.imwrite(output_variable,recorrection_image)

print("PROGRAM ENDED")
print('-------------------------------------------------------------------------')
