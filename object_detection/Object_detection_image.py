######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description:
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
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
parser.add_argument("-o", "--output", type=str, help="path to optional output image file")  ### None
parser.add_argument("-wr", "--wheelchair_rear", type=float, default=0.8, help="size of wheelchair rear in meters")
parser.add_argument("-ws", "--wheelchair_side", type=float, default=1.2, help="size of wheelchair side in meters")
parser.add_argument("-c", "--threshold", type=float, default=0.8, help="minimum probability to filter weak detection")
parser.add_argument("-d", "--distance", type=float, default=0.7, help="segmentation percentage for distance criteria")
args = parser.parse_args()
vargs = vars(args)

def PolyArea(x,y):  # SHOELACE FORMULA
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(copy_image,(x,y),6,(0,255,255),-1)
        mouseX,mouseY = x,y

drawing = False # true if mouse is pressed
ix,iy = -1,-1
# mouse callback function
def draw_square(event,x,y,flags,param):
    global ix,iy,xx,yy,drawing
    if x<0:
        x = 0
    elif y<0:
        y = 0
    elif x<0 and y<0:
        x,y = 0,0

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

if vargs["output"]==None:
    output_variable = 'output.jpg'
else:
    output_variable = vargs["output"]

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,vargs["model"])#MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,vargs["labels"])#'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,vargs["input"])#IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
copy_image = image.copy()
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

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

boxes = boxes.reshape(boxes.shape[1],4)
scores = scores.reshape(scores.shape[1],1)
classes = classes.reshape(classes.shape[1],1)

# Draw the results of the detection (aka 'visulaize the results')
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
    elif (puntaje>min_score) and (clase==2.0):
        x = np.array([caja[1],caja[1]+distance_x,caja[1]+distance_x,caja[1]])*image.shape[1]
        y = np.array([caja[0],caja[0],caja[0]+distance_y,caja[0]+distance_y])*image.shape[0]
        vis_util.draw_bounding_box_on_image_array(copy_image,caja[0],caja[1],caja[2],caja[3],color='yellow',thickness=4,display_str_list=()) # WHEELCHAIRS
        wheelchair_area = PolyArea(x,y)

print('People detected =',people)
wheelchair_rear = vargs["wheelchair_rear"]
wheelchair_side = vargs["wheelchair_side"]
standard_wheelchair_area = wheelchair_side*wheelchair_rear
print('Wheelchair area =', standard_wheelchair_area, 'm2')
cv2.circle(copy_image,(int(centroid[0]),int(centroid[1])),6,(0,255,255),-1)
polygon_area = PolyArea(pts[:,0],pts[:,1])
pts = pts.reshape((-1,1,2))
cv2.polylines(copy_image,[pts],True,(0,255,255))
cv2.imshow('Area selection',copy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

input1 = input('Handcrafted detections?' )
cv2.namedWindow('Handcrafted image')
cv2.setMouseCallback('Handcrafted image',draw_square)
agregate = 0
while(input1=='y'):
    cv2.imshow('Handcrafted image',image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('a'): # agrega cabezas
        boxes = np.concatenate((boxes,np.array([[iy/image.shape[0],ix/image.shape[1],yy/image.shape[0],xx/image.shape[1]]])))
        print(np.array([[iy/image.shape[0],ix/image.shape[1],yy/image.shape[0],xx/image.shape[1]]]))
        scores = np.concatenate((scores,np.array([[0.99]])))
        classes = np.concatenate((classes,np.array([[1.0]])))
        agregate-=1
    elif k == ord('q'): # agrega sillas
        classes[classes==2.0]=0.0
        boxes = np.concatenate((boxes,np.array([[iy/image.shape[0],ix/image.shape[1],yy/image.shape[0],xx/image.shape[1]]])))
        scores = np.concatenate((scores,np.array([[0.99]])))
        classes = np.concatenate((classes,np.array([[2.0]])))
cv2.destroyAllWindows()

people = 0
for puntaje,caja,clase in zip(scores,boxes,classes):
    distance_x = caja[3] - caja[1]
    distance_y = caja[2] - caja[0]
    point = np.array([(caja[1]+distance_x)*image.shape[1],(caja[0]+distance_y)*image.shape[0]])
    dist = np.linalg.norm(centroid-point)
    if (puntaje>min_score) and (clase==1.0) and (dist<=min_length*vargs["distance"]):
        people+=1
        vis_util.draw_bounding_box_on_image_array(copy_image,caja[0],caja[1],caja[2],caja[3],color='red',thickness=4,display_str_list=()) # HEADS
    elif (puntaje>min_score) and (clase==2.0):
        x = np.array([caja[1],caja[1]+distance_x,caja[1]+distance_x,caja[1]])*image.shape[1]
        y = np.array([caja[0],caja[0],caja[0]+distance_y,caja[0]+distance_y])*image.shape[0]
        vis_util.draw_bounding_box_on_image_array(copy_image,caja[0],caja[1],caja[2],caja[3],color='yellow',thickness=4,display_str_list=()) # WHEELCHAIRS
        wheelchair_area = PolyArea(x,y)

if agregate != 0:
    for caja in boxes[agregate:]:
        people+=1
        vis_util.draw_bounding_box_on_image_array(copy_image,caja[0],caja[1],caja[2],caja[3],color='red',thickness=4,display_str_list=())

total_area = float(polygon_area/wheelchair_area*standard_wheelchair_area)
people_density = float(people/total_area)
print('People detected =',people)
print('Measurement Area =', total_area, 'm2')
print('Density = ', people_density, 'people/m2')
print('------------------------------------------------------------------------')

cv2.imshow('Image',copy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

iter = 1
input2 = input('Wheelchair area correction?') ### parser porcentaje
dec = True
while(input2=='y'):
    input1 = float(input('Correction Value (percentage)?'))
    for puntaje,caja,clase in zip(scores,boxes,classes):
        distance_x = (caja[3] - caja[1])
        distance_y = caja[2] - caja[0]
        point = np.array([(caja[1]+distance_x)*image.shape[1],(caja[0]+distance_y)*image.shape[0]])
        dist = np.linalg.norm(centroid-point)
        if (puntaje>min_score) and (clase==1.0) and (dist<=min_length*vargs["distance"]):
            vis_util.draw_bounding_box_on_image_array(correction_image,caja[0],caja[1],caja[2],caja[3],color='red',thickness=4,display_str_list=()) # HEADS
        elif (puntaje>min_score) and (clase==2.0):
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

    print('Wheelchair area =', standard_wheelchair_area, 'm2')
    total_area = float(polygon_area/wheelchair_area*standard_wheelchair_area)
    people_density = float(people/total_area)
    print('Measurement area =', total_area, 'm2')
    print('Density = ', people_density, 'people/m2' )
    print('--------------------------------------------------------------------')
    iter+=1
    dec = False
    input2 = input('Do you need to make a new correction?')

# final picture
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

print("Program ended")
