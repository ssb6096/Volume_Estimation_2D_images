import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
img = cv2.imread('D:\data_sequences\data_sequences\\01\\rs435i_1\\rgb\\0000000180.jpg')



#Using Object detection YOLO to identify the food. Did not work. Identified the laptop instead
#If there is a good food detection function, call here-
#Can use the bounding box to find the area.
#Can get the co-ordinates of the contour identified as food
bbox, label, conf = cv.detect_common_objects(img)
output_image = draw_bbox(img, bbox, label, conf)
plt.imshow(output_image)
plt.show()
print("bbox")
print(bbox)
print("conf")
print(conf)