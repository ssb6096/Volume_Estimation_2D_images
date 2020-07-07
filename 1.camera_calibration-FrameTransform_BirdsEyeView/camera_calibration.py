"""
This code assumes that images used for calibration are of the same arUco marker board provided with code

"""

import cv2
from cv2 import aruco
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm


import cvlib as cv
from cvlib.object_detection import draw_bbox
import matplotlib.pyplot as plt
import imutils
import PIL
import matplotlib as mpl
from scipy.spatial import ConvexHull

# root directory of repo for relative path specification.
root = Path(__file__).parent.absolute()

# Set this flsg True for calibrating camera and False for validating results real time
calibrate_camera = True

# Set path to the images
calib_imgs_path = root.joinpath("aruco_data")

# For validating results, show aruco board to camera.
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )

#Provide length of the marker's side
markerLength = 3.75  # Here, measurement unit is centimetre.

# Provide separation between markers
markerSeparation = 0.5   # Here, measurement unit is centimetre.

column_markers=3
row_markers=3
# create arUco board
board = aruco.GridBoard_create(column_markers, row_markers, markerLength, markerSeparation, aruco_dict)




fig = plt.figure()
nx = 3
ny = 3
for i in range(1, nx*ny+1):
    ax = fig.add_subplot(ny,nx, i)
    img = aruco.drawMarker(aruco_dict,i, 700)
    plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")

plt.savefig("markers.pdf")
plt.show()
plt.close()
#print(board)
#print(type(board))

'''height=int((markerLength*row_markers+markerSeparation*(row_markers+1))*(37.795275591))
width=int((markerLength*column_markers+markerSeparation*(column_markers+2))*(37.795275591))
'''
#margin=10
#height=int((markerLength*row_markers+markerSeparation*(row_markers)-markerSeparation+2*margin)*(26.795275591))
#width=int((markerLength*column_markers+markerSeparation*(column_markers)-markerSeparation+2*margin)*(26.795275591))
#print(height,width)
'''uncomment following block to draw and show the board'''
#borderBits=0
#img = board.draw((700,700)) #1 pixel=0.0264cm
'''imageSize.width = markersX * (markerLength + markerSeparation) - markerSeparation + 2 * margins;
    imageSize.height =
        markersY * (markerLength + markerSeparation) - markerSeparation + 2 * margins;

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Ptr<aruco::GridBoard> board = aruco::GridBoard::create(markersX, markersY, float(markerLength),
                                                      float(markerSeparation), dictionary);

    // show created board
    Mat boardImage;
    board->draw(imageSize, boardImage, margins, borderBits);

    if(showImage) {
        imshow("board", boardImage);
        waitKey(0);
    }

    imwrite(out, boardImage);'''
#cv2.imshow("aruco", img)
#cv2.waitKey(0)
#cv2.imwrite("aruco.jpg",img)
arucoParams = aruco.DetectorParameters_create()

if calibrate_camera == True:
    img_list = []
    calib_fnms = calib_imgs_path.glob('*.jpg')
    print('Using ...', end='')
    for idx, fn in enumerate(calib_fnms):
        print(idx, '', end='')
        img = cv2.imread( str(root.joinpath(fn) ))
        img_list.append( img )
        h, w, c = img.shape
    print('Calibration images')

    counter, corners_list, id_list = [], [], []
    first = True
    for im in tqdm(img_list):
        img_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        if first == True:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
        counter.append(len(ids))
    print('Found {} unique markers'.format(np.unique(ids)))

    counter = np.array(counter)
    print(counter)
    print ("Calibrating camera .... Please wait...")
    #mat = np.zeros((3,3), float)
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

    print("Camera matrix is \n", mtx, "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open("calibration.yaml", "w") as f:
        yaml.dump(data, f)

else:
    camera = cv2.VideoCapture(0)
    ret, img = camera.read()

    with open('calibration.yaml') as f:
        loadeddict = yaml.load(f)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)

    ret, img = camera.read()
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    h,  w = img_gray.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    pose_r, pose_t = [], []
    while True:
        ret, img = camera.read()
        img_aruco = img
        im_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        h,  w = im_gray.shape[:2]
        dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)
        #cv2.imshow("original", img_gray)
        if corners == None:
            print ("pass")
        else:

            ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist) # For a board
            print ("Rotation ", rvec, "Translation", tvec)
            if ret != 0:
                img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
                img_aruco = aruco.drawAxis(img_aruco, newcameramtx, dist, rvec, tvec, 10)    # axis length 100 can be changed according to your requirement

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break;
        cv2.imshow("World co-ordinate frame axes", img_aruco)
        cv2.imwrite('D:\Deep Learning\project_code\camera_calibration-master\camera_calibration-master\messigray.png', img_aruco)
cv2.destroyAllWindows()

#https://www.learnopencv.com/augmented-reality-using-aruco-markers-in-opencv-c-python/
#Could be used for the food detection
#We know the size of the marker and where its placed
#img = cv2.imread("D:\Deep Learning\project_code\camera_calibration-master\camera_calibration-master\WhatsApp Image 2020-04-05 at 11.41.42 AM.jpg")
#img = cv2.imread("D:\Deep Learning\project_code\camera_calibration-master\camera_calibration-master\WhatsApp Image 2020-04-05 at 1.14.16 PM.jpeg")
#img = cv2.imread("D:\Deep Learning\project_code\camera_calibration-master\camera_calibration-master\WhatsApp Image 2020-04-05 at 2.51.27 PM.jpeg")
img = cv2.imread("D:\Deep Learning\project_code\camera_calibration-master\camera_calibration-master\\New Folder\\4.jpeg")
print(img)
im_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
allCorners, allIds, rejectedImgPoints = aruco.detectMarkers(im_gray, aruco_dict, parameters=arucoParams)
frame_markers = aruco.drawDetectedMarkers(img.copy(), allCorners, allIds)
plt.figure()
plt.imshow(frame_markers)
plt.legend()
plt.show()


#allCorners, allIds, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)
'''for element in allCorners:
    print(element.size)
    print(element)
print(allIds)'''
#The commented out code would be to get the extrinsic parameters during camera calibration
#flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL)
'''flags=0
corners_list=allCorners
ids_list=allIds
counter=20'''
'''(ret, camera_matrix, distortion_coefficients0,
 rotation_vectors, translation_vectors,
 stdDeviationsIntrinsics, stdDeviationsExtrinsics,
 perViewErrors) = cv2.aruco.calibrateCameraArucoExtended(
    corners=corners_list,
    ids=ids_list,
    counter=20,
    board=board,
    imageSize=im_gray.shape,
    cameraMatrix=mtx,
    distCoeffs=dist,
    flags=0,
    criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))'''
'''flags=flags,
    criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))'''
#matrix = cv2.getPerspectiveTransform(allCorners, board)
#ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(allCorners, allIds, counter, board, im_gray.shape, None, None )
#To get the extrinsic parameters after camera calibration - use SolvePnP- cant
rvec= np.empty(shape=(1,3))
tvec=np.empty(shape=(3,1))
trash,rvec, tvec= aruco.estimatePoseBoard(allCorners,allIds,board, mtx, dist,rvec=rvec,tvec=tvec) # For all markers marker
print("rvec")
print(rvec)
print("tvec")
print(tvec)
print(len(tvec))

length_of_axis = 10
imaxis = aruco.drawDetectedMarkers(img.copy(), allCorners, allIds)
#for i in range(len(tvec)):
imaxis = aruco.drawAxis(imaxis, mtx, dist, rvec, tvec, length_of_axis)
plt.figure()
plt.imshow(imaxis)
plt.show()
corners, ids, rejectedImgPoints = aruco.detectMarkers(im_gray, aruco_dict, parameters=arucoParams)
size_of_marker=0.0375
rvecs,tvecs, trash = aruco.estimatePoseSingleMarkers(allCorners, size_of_marker, mtx, dist)
length_of_axis = 0.01
imaxis = aruco.drawDetectedMarkers(img.copy(), corners, ids)
for i in range(len(tvecs)):
    imaxis = aruco.drawAxis(imaxis, mtx, dist, rvecs[i], tvecs[i], length_of_axis)
plt.figure()
plt.imshow(imaxis)
plt.show()
'''(_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coefs)'''
#Using Object detection YOLO to identify the food. Did not work. Identified the laptop instead
#If there is a good food detection function, call here-
#Can use the bounding box to find the area.
#Can get the co-ordinates of the contour identified as food
'''bbox, label, conf = cv.detect_common_objects(img)
output_image = draw_bbox(img, bbox, label, conf)
plt.imshow(output_image)
plt.show()
print("bbox")
print(bbox)
print("conf")
print(conf)'''
#Using Computer vision techniques
#ret, thresh = cv2.threshold(im_gray, 127, 255, 0)
'''ret, thresh = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#contours,hierarchy = cv2.findContours(thresh, 1, 2)
cv2.drawContours(img, contours, -1, (0,255,0), 3)
print(contours)
print("Number of Contours found = " + str(len(contours)))
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
#Too many circles were detected
'''img = cv2.medianBlur(im_gray,5)
cimg = cv2.cvtColor(im_gray,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
#Now what ??
# Blurring for removing the noise
img_blur = cv2.bilateralFilter(img, d = 7,
                               sigmaSpace = 75, sigmaColor =75)
# Convert to grayscale
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
# Apply the thresholding
a = img_gray.max()
_, thresh = cv2.threshold(img_gray, a/2+10, a,cv2.THRESH_BINARY_INV)
plt.imshow(thresh, cmap = 'gray')
# Find the contour of the figure
contours, hierarchy = cv2.findContours(
                                   image = thresh,
                                   mode = cv2.RETR_TREE,
                                   method = cv2.CHAIN_APPROX_SIMPLE)
# Sort the contours
contours = sorted(contours, key = cv2.contourArea, reverse = True)
# Draw the contour
img_copy = img.copy()
cv2.drawContours(img_copy, contours, contourIdx = -1,
                         color = (255, 0, 0), thickness = 2)
'''cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)'''
#plt.imsave("IMG.JPEG",img_copy)
#plt.imsave(final)
print("Number of Contours found = " + str(len(contours)))
'''for c in contours:
    print(c[1][0][1])
    break'''
#contours = imutils.grab_contours(contours)
i=0
for c in contours:
	i=i+1
	# draw the contour and center of the shape on the image
	cv2.drawContours(img_copy, [c], -1, (0, 255, 0), 2)
	cv2.putText(img_copy, str(i), (c[0][0][0], c[0][0][1]),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow('Contours', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(contours[4])
distanceCoefficients=3.75
retval, rvec, tvec = aruco.estimatePoseBoard(allCorners, allIds, board, mtx,dist,rvecs[1],tvec)
print("rvec")
print(rvec)
dst,_ = cv2.Rodrigues(rvec)
print("dst")
print(dst)
extristics = [[dst[0][0],dst[0][1],dst[0][2],tvec[0][0]],
                             [dst[1][0],dst[1][1],dst[1][2],tvec[1][0]],
                             [dst[2][0],dst[2][1],dst[2][2],tvec[2][0]],
                             [0.0, 0.0, 0.0, 1.0]]
extristics_I = np.linalg.inv(extristics) # inverse matrix
worldPos = [extristics_I[0,3],extristics_I[1,3],extristics_I[2,3]]
print("extristics")
print(extristics)
print("world co-ordinate position")
print(worldPos)

def groundProjectPoint(image_point, z = 0.0):
    camMat = np.asarray(mtx)
    iRot = np.linalg.inv(dst)
    iCam = np.linalg.inv(camMat)

    uvPoint = np.ones((3, 1))

    # Image point
    uvPoint[0, 0] = image_point[0]
    uvPoint[1, 0] = image_point[1]

    tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
    tempMat2 = np.matmul(iRot, tvec)

    s = (z + tempMat2[2, 0]) / tempMat[2, 0]
    wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvec))

    # wcPoint[2] will not be exactly equal to z, but very close to it
    assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
    wcPoint[2] = z

    return wcPoint
i=0
#gpp=np.array()
gpp=[]
for c in contours[4]:
    pixel = (c[0][0],c[0][1])
    print("Pixel: %s" % (pixel, ))
    '''gpp[i][0][0]=groundProjectPoint(pixel)[0]
    gpp[i][0][1] = groundProjectPoint(pixel)[1]
    gpp[i][0][2] = groundProjectPoint(pixel)[2]'''
    g=groundProjectPoint(pixel)
    gpp.append(g)
    print(groundProjectPoint(pixel))
    i=i+1


points=np.array(gpp)
points=np.reshape(points,(482,3),order='C')
print(points)
print(len(points))
print(len(points[0]))

max=np.amax(points, axis=0)

min=np.amin(points, axis=0)

#volume = (x_max-x_min)*(y_max-y_min)*(z_max-z_min)
print(max)
print(min)
diff=max-min
print(diff)
print(diff[0]*diff[1])

#Birds Eye view
'''
IMAGE_H = 223
IMAGE_W = 1280

src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

img = cv2.imread('./test_img.jpg') # Read the test img
img = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
plt.show()
'''
# Set of destinations points for "birds eye view"
# dimension of the new image
'''
maxWidth=img.shape[0]
maxHeight=img.shape[1]
dst = np.array([
    [0,0],
    [maxWidth-1, 0],
    [maxWidth-1, maxHeight-1],
    [0, maxHeight-1]], dtype="float32")
# compute the perspective transform matrix and then apply it
transformMatrix = cv2.getPerspectiveTransform(allCorners[0], dst)
# transform ROI
scan = cv2.warpPerspective(img, transformMatrix, (maxWidth*10, maxHeight*10))
# lets see the wraped document
cv2.imshow("Scaned",scan)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#Perspective wrap and other methods to obtain birds eye view
img1 = cv2.imread("D:\Deep Learning\project_code\camera_calibration-master\camera_calibration-master\\New Folder\\5.jpeg")
print(img1)
im_gray1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
allCorners1, allIds1, rejectedImgPoints1 = aruco.detectMarkers(im_gray1, aruco_dict, parameters=arucoParams)
# Blurring for removing the noise
img_blur1 = cv2.bilateralFilter(img1, d = 7,
                               sigmaSpace = 75, sigmaColor =75)
# Convert to grayscale
img_gray1 = cv2.cvtColor(img_blur1, cv2.COLOR_RGB2GRAY)
# Apply the thresholding
a1 = img_gray1.max()
_, thresh1 = cv2.threshold(img_gray1, a1/5, a1,cv2.THRESH_BINARY_INV)
plt.imshow(thresh, cmap = 'gray')
# Find the contour of the figure
contours1, hierarchy1 = cv2.findContours(
                                   image = thresh1,
                                   mode = cv2.RETR_TREE,
                                   method = cv2.CHAIN_APPROX_SIMPLE)
# Sort the contours
contours1 = sorted(contours1, key = cv2.contourArea, reverse = True)
cv2.drawContours(img1, contours1, contourIdx = -1,
                         color = (255, 0, 0), thickness = 2)
cv2.imshow("AllShown",img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
i=0
for c in contours1:
	i=i+1
	# draw the contour and center of the shape on the image
	cv2.drawContours(img1, [c], -1, (0, 255, 0), 2)
	cv2.putText(img1, str(i), (c[0][0][0], c[0][0][1]),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow('Contours', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

x,y,w,h = cv2.boundingRect(contours1[2])
print(x,y,w,h)
#cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
#cv2.imshow("Show",img1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

x2=x+w
y2=y+h
srcPoints=np.array([[x,y],[x2,y],[x2,y2],[x,y2]])

img2 = cv2.imread("D:\Deep Learning\project_code\camera_calibration-master\camera_calibration-master\\New Folder\\6.jpeg")
print(img2)
im_gray2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
allCorners2, allIds2, rejectedImgPoints2 = aruco.detectMarkers(im_gray2, aruco_dict, parameters=arucoParams)
dstPoints=allCorners2[1]
src=allCorners1[1]
print(allCorners2[1])
print(type(allCorners2))
print(len(allCorners2))
#print(allCorners2.shape)
h=cv2.findHomography(srcPoints, dstPoints)[0]
print("h")
print(h)
img_out=cv2.warpPerspective(im_gray1, h,(img2.shape[1],img2.shape[0]))
cv2.imshow("Warped Source Image", img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()


maxWidth=img1.shape[0]
maxHeight=img1.shape[1]
dst = np.array([
    [0,0],
    [maxWidth-1, 0],
    [maxWidth-1, maxHeight-1],
    [0, maxHeight-1]], dtype="float32")
# compute the perspective transform matrix and then apply it
transformMatrix = cv2.getPerspectiveTransform(src, dst)
# transform ROI
scan = cv2.warpPerspective(img1, transformMatrix, (maxWidth*10, maxHeight*10))
# lets see the wraped document
cv2.imshow("Scaned",scan)
cv2.waitKey(0)
cv2.destroyAllWindows()