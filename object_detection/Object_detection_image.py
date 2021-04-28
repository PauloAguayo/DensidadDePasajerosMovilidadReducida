import cv2
import numpy as np
import tensorflow as tf
from shapely import geometry
import argparse
from measurements import Measurements
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from draw import Drawing
from outcomes import Outcomes
from calibrate import Calibration
import csv
import os
from scipy.spatial import ConvexHull

# Parser arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", required=True, help="path to object detection model")
parser.add_argument("-l", "--labels", required=True, help="path to labels file")
parser.add_argument("-i", "--input", default=0, type=str, help="path to optional input image file", required=True)
parser.add_argument("-o", "--output", type=str, default="results/output.jpg", help="path and name to optional output image file")
parser.add_argument("-c", "--threshold", type=float, default=0.8, help="minimum probability to filter weak detection")
parser.add_argument("-wr", "--wheelchair_rear", type=float, default=0.85, help="size of wheelchair rear in meters")
parser.add_argument("-ws", "--wheelchair_side", type=float, default=1.2, help="size of wheelchair side in meters")
parser.add_argument("-t", "--calibration", action="store_true", help="option for un-distort input image")
parser.add_argument("-r", "--resize", type=str, default="720,1280", help="resize input image")
parser.add_argument("-H", "--camera_height", type=float, default=2.5, help="z-coordinate for camera positioning")
parser.add_argument("-p", "--people_height", type=float, default=1.7, help="z-coordinate for people high")
parser.add_argument("-a", "--angle", type=float, default=15, help="positioning angle in degrees")
parser.add_argument("-mode", "--mode", type=int, required=True, help="selection mode: 1 = wheelchair ground truth; 2 = polygon ground truth")
args = parser.parse_args()
vargs = vars(args)

# Calling objects
mode = Outcomes(vargs["mode"],vargs["wheelchair_rear"],vargs["wheelchair_side"])
intro = mode.Mode()
txt_intro = mode.txt()
measures = Measurements(vargs["mode"])

# Defining path and name of output picture
output_variable = vargs["output"]

# Storing paths
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,vargs["model"])
PATH_TO_LABELS = os.path.join(CWD_PATH,vargs["labels"])
PATH_TO_IMAGE = os.path.join(CWD_PATH,vargs["input"])

# Storing heights and angle
people_height = vargs["people_height"]
camera_height = vargs["camera_height"]
angle = float(np.pi*vargs["angle"]/180)


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

# Resizing image
resized = (int(vargs["resize"].split(',')[0]),int(vargs["resize"].split(',')[1]))

# Loading image
image = cv2.imread(PATH_TO_IMAGE)
image = cv2.resize(image, (resized[1],resized[0]), interpolation = cv2.INTER_AREA)

# Undistort images
if vargs["calibration"] == True:
    calibration = Calibration([resized[1],resized[0]])
    calibration.Checkboard()
    balance = 1
    image_calib = image.copy()
    image = calibration.Undistort(image,balance)
    cv2.imshow('Undistorted',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ask = input('DO YOU WANT TO CORRECT THE UNDISTORTED IMAGE (y/n)?')
    while ask=='y':
        print('balance = '+str(balance))
        balance = float(input('BALANCE VALUE? '))
        image = calibration.Undistort(image_calib,balance)
        cv2.imshow('Undistorted',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ask = input('CORRECT BALANCE VALUE? (y/n)')

# Auxiliary copies
copy_image = image.copy()
final_image = image.copy()

# Expanding image dimensions to have shape: [1, None, None, 3]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# Object to draw detections
draws = Drawing(vargs["mode"],image,copy_image,final_image,vargs["threshold"],angle)

# Get polygon and centroid points
pts = np.array(draws.Generate_Polygon('Image '+vargs["input"]), np.int32)
poly = geometry.Polygon(pts)
centroid = np.array(list(poly.centroid.coords)[0])

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

draws.Prepare_data(scores,boxes,classes)

# First detections
detections_1 = draws.Draw_detections(0,0,camera_height,people_height)
print('-------------------------------------------------------------------------')
print('PEOPLE DETECTED =',detections_1[1])

# Drawing polygon and its centroid
cv2.circle(copy_image,(int(centroid[0]),int(centroid[1])),6,(0,255,255),-1)
#polygon_area = measures.PolyArea(pts[:,0],pts[:,1])
pts = pts.reshape((-1,1,2))
cv2.polylines(copy_image,[pts],True,(0,255,255))
cv2.polylines(image,[pts],True,(0,255,255))
cv2.imshow('Area selection',copy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Auxiliary function
mode.Extra_dec()

# Some calculations
#people_density = mode.People_density(detections_1[1],polygon_area,detections_1[0])
#print(txt_intro, people_density[1], 'm2')
#print('DENSITY = ', people_density[0], 'people/m2')
print('------------------------------------------------------------------------')

# In case detections are incomplete or not perfect, you can make square selections to ensure a better performance
input1 = input('HANDCRAFTED SELECTIONS? (y/n)' )
print('-------------------------------------------------------------------------')
if input1=='y':
    draws.Handcrafted(input1)

# 2nd detections (refinement)
detections_2 = draws.Draw_detections(0,1,camera_height,people_height)

people = int(detections_2[1])
polygon_area = draws.Voronoi_diagram(image)
cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2nd calculations (refinement)
print('PEOPLE DETECTED =',people)
people_density = mode.People_density(people,polygon_area,detections_2[0])
print(txt_intro, people_density[1], 'm2')
print('AVERAGE DENSITY = ', people_density[0], 'people/m2')
print('------------------------------------------------------------------------')

# Loop for bounding box correction for wheelchair detection following the same last procedure for plotting squares
iter = 1
input2 = input('WHEELCHAIR AREA CORRECTION? (y/n)') # Percentage input
print('-------------------------------------------------------------------------')
dec = True
while(input2=='y'):
    input1 = float(input('CORRECTION VALUE? (PERCENTAGE) '))
    print('-------------------------------------------------------------------------')
    detections_3 = draws.Draw_detections(input1,1,camera_height,people_height) #(scores,boxes,classes,centroid,input1,n)

    cv2.imshow('Correction n_'+str(iter),image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('PEOPLE DETECTED =',people)
    people_density = mode.People_density(people,polygon_area,detections_3[0]) #(people,polygon_slf,chair_slf)
    print(txt_intro, people_density[1], 'm2')
    print('AVERAGE DENSITY = ', people_density[0], 'people/m2')
    print('--------------------------------------------------------------------')
    iter+=1
    dec = False
    input2 = input('DO YOU NEED A NEW CORRECTION? (y/n)')
    print('-------------------------------------------------------------------------')

# Write a CSV file with the most recent calculations
with open(output_variable.split('.')[0]+'.csv','w',newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['PEOPLE DETECTED',txt_intro.split('=')[0]+' (m2)','DENSITY (people/m2)'])
    csv_writer.writerow([people,people_density[1],people_density[0]])

# Getting the final picture to save it as the "output_variable" for practical and analytical uses
if dec == True: input1 = 0
draws.Draw_detections(input1,2,camera_height,people_height)

cv2.circle(final_image,(int(centroid[0]),int(centroid[1])),6,(0,255,255),-1)
cv2.polylines(final_image,[pts],True,(0,255,255))
cv2.imwrite(output_variable,final_image)

print("PROGRAM FINISHED")
print('-------------------------------------------------------------------------')
