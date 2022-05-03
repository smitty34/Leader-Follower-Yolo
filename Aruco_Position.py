
import cv2
from cv2 import aruco
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R


def readCameraParams():
    
    with open(path_camera_params, "r") as file:
        cam_params = json.load(file)
        
        # Camera Matrix
        mtx = np.array(cam_params["mtx"])
        
        # distortion coefficients
        dist = np.array(cam_params["dist"])
        
        return mtx, dist
    


def readCentroidData(path_dcnn_data):
    #open data file with centroids and bboxes from DCNN detection and store it in centroid_data variable
    centroid_data = []    
    
    with open(path_dcnn_data) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 1:
                temp = []
                for i in range(17):
                    if row[i] == '' or row[i] == 'nan':
                        row[i] = 0
                    temp.append(int(row[i]))
                centroid_data.append(temp)
            line_count += 1    
    csv_file.close()
    
    return centroid_data




def Aruco_Parameters():
    parameters = aruco.DetectorParameters_create()
    
    parameters.minMarkerPerimeterRate = 0.01
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.33
    parameters.errorCorrectionRate = 2.0
    parameters.aprilTagMinClusterPixels = 100
    parameters.aprilTagMaxNmaxima = 5
    parameters.aprilTagCriticalRad = 20*np.pi/180
    parameters.aprilTagMaxLineFitMse = 1
    parameters.aprilTagMinWhiteBlackDiff = 100
    
    return parameters



def setMarker_size():
    avg_size = np.zeros((1,1))
    
    return avg_size


def preprocessFrame(frame):
    
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    
    lab[...,0] = cv2.LUT(lab[...,0], lookUpTable)
    
    frame = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    
    return frame




def detect_Aruco(gray, parameters):
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=mtx, distCoeff=dist)
    
    return corners, ids
    
    
    

gamma = 2 # gamma parameter value
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i/255.0, gamma)* 255.0, 0, 255)
    
    
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (width, height), 5)




def detectArucoMarkers(gray, parameters):
    #use predefined Aruco markers dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    
    #detect markers with APRILTAG method
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=mtx, distCoeff=dist)

    return corners, ids

def getMarkerData(corners, rvec, cx_prev, cy_prev):
    #marker centre x and y
    cx = int(corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4
    cy = int(corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4
    
    #marker size in pixels, cosine of yaw angle, sine of yaw angle
    msp = ((np.sqrt(np.power((corners[1][0]-corners[0][0]),2) + np.power((corners[1][1]-corners[0][1]),2)) + 
               np.sqrt(np.power((corners[2][0]-corners[1][0]),2) + np.power((corners[2][1]-corners[1][1]),2)) + 
               np.sqrt(np.power((corners[3][0]-corners[2][0]),2) + np.power((corners[3][1]-corners[2][1]),2)) + 
               np.sqrt(np.power((corners[0][0]-corners[3][0]),2) + np.power((corners[0][1]-corners[3][1]),2))) / 4)
        
    #distance in metres between marker of the same ID on subsequent frames
    if cx_prev is not None and cy_prev is not None:
        diff = np.sqrt(np.power(cx_prev-cx,2) + np.power(cy_prev-cy,2)) * markerLength / msp
    else:
        diff = 0
    
    return abs(cx), abs(cy), msp, diff



def calculateAverageMarkerSize(msp_avg, msp):
    #write last measured marker size into table
    if(N_avg == 1):
        msp_avg = msp
    elif(N_avg > 1 and isinstance(N_avg, int)):
        for j in range(N_avg-1):
            msp_avg[j] = msp_avg[j+1]
        msp_avg[N_avg-1] = msp
    
    #calculate the average and rescale marker size
    nonzero = np.count_nonzero(msp_avg)
    size_corr = np.sum(msp_avg)/(msp*nonzero)
    msp = msp * size_corr
    
    return size_corr, msp



def markerLengthCorrection(attitude):
    #use correction of marker size based on current altitude
    return markerLengthOrg * (1 - 0.00057 * attitude/marker_div) / div



def printDataOnImage(corners, tvec, rvec, ids):
    font = cv2.FONT_HERSHEY_SIMPLEX
    r = R.from_rotvec(rvec)    
    
    #calculate real altitude to be printed
    tvec_temp = tvec.copy()
    tvec_temp[2] = tvec_temp[2]/marker_div
    
    #calculate angles and position and convert them to text
    ang = 'R = ' + str([round(r.as_euler('zxy', degrees=True)[0],2),
                        round(r.as_euler('zxy', degrees=True)[1],2),
                        round(r.as_euler('zxy', degrees=True)[2],2)]) + 'deg' 
    pos = 't = ' + str([round(j,3) for j in tvec_temp]) + 'm'  
    id = 'ID = ' + str(ids)    
    
    #calculate the position where the text will be placed on image
    position = tuple([int(corners[0]-150), int(corners[1]+150)])
    position_ang = tuple([int(position[0]-0), int(position[1]+50)])
    position_id = tuple([int(position[0]-0), int(position[1]-50)])
    
    #write the text onto the image
    cv2.putText(frame, id, position_id, font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, pos, position, font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, ang, position_ang, font, 1.4, (255, 255, 255), 2, cv2.LINE_AA)




def boundingBoxFromDCNN(centroid_data_x, centroid_data_y):
    #use the closest point of the vehicle from DCNN detection
    xc = centroid_data_x
    yc = centroid_data_y
    imgpts = np.maximum(0,np.int32(np.array([[xc, yc, 0]])))
    if drawPoints:
        cv2.circle(frame, (int(imgpts[0][0]),int(imgpts[0][1])), 5, color=(255,255,255), thickness=-1)
    
    return imgpts

def drawBoundingBox(tvec, rvec, veh_dim, size_corr):
    #calculate angles in horizontal and vertical direction
    alpha_h = np.arctan(tvec[0]/tvec[2])
    alpha_v = np.arctan(tvec[1]/tvec[2])
    
    #calucalate yaw angle of the vehicle
    r = R.from_rotvec(rvec[0])
    yaw = round(r.as_euler('zxy', degrees=True)[0],2)
    
    #based on yaw angle of the vehicle, alpha angles may be negative
    alpha_h = alpha_h if yaw < 0 else -alpha_h
    alpha_v = alpha_v if yaw < 0 else -alpha_v
    
    #modify dimensions of vehicle's bbox
    veh_dim = np.multiply(veh_dim, [1-alpha_h/2, 1+alpha_h/2, 1-alpha_v/2, 1+alpha_v/2])
    
    #use modified values to set corners of bbox, project these points onto the image and draw bbox
    axis = np.float32([[veh_dim[2],veh_dim[0],0], [veh_dim[2],veh_dim[1],0], [veh_dim[3],veh_dim[1],0], [veh_dim[3],veh_dim[0],0]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec/size_corr, mtx, dist)
    imgpts = np.maximum(0,np.int32(imgpts).reshape(-1,2))
    cv2.drawContours(frame, [imgpts[0:4]], -1, (255,0,0), 5)

    return veh_dim





def generatePointsBoundingBox(veh_dim):
    #generate additional points on bounding box - 20 along the length and 8 along the width of the vehicle
    points_l = 20
    points_w = 8
    
    o1 = np.linspace(veh_dim[0], veh_dim[1], points_l)
    o2 = np.linspace(veh_dim[2], veh_dim[3], points_w)
    
    object1 = np.zeros((points_l,2))
    object2 = np.zeros((points_l,2))
    object3 = np.zeros((points_w,2))
    object4 = np.zeros((points_w,2))
    
    object1[:,0] = o1
    object1[:,1] = veh_dim[2]
    object2[:,0] = o1
    object2[:,1] = veh_dim[3]
    object3[:,0] = veh_dim[0]
    object3[:,1] = o2
    object4[:,0] = veh_dim[1]
    object4[:,1] = o2
    
    #concatenate the points generated on each edge of bbox    
    object = np.concatenate((object1, object2, object3, object4))
    w, h = object.shape
    bbox = np.zeros((w, h+1))
    
    bbox[:,0] = object[:,1]
    bbox[:,1] = object[:,0]
    bbox[:,2] = 0
    
    return bbox



def findMinimumDistanceBoundingBox(source, bbox, tvec, rvec, size_corr):
    #project generated bbox points onto image
    imgpts, _ = cv2.projectPoints(bbox, rvec, tvec/size_corr, mtx, dist)
    imgpts = np.maximum(0,np.int32(imgpts).reshape(-1,2))

    #find minimum distance between source of signal and generated bbox points
    distance = np.inf
    index = 0
    for i in range(len(imgpts)):
        d = np.sqrt(pow(source[0][0]-imgpts[i][0],2) + pow(source[0][1]-imgpts[i][1],2))
        if(d < distance):
            distance = d
            index = i
    
    #return the closest point
    return imgpts[index]


def calculateDistance(lidar, aruco, bbox, markerLength, msp4, msp):
    #calculate distances to Aruco marker and bbox of the vehicle
    d_aruco = np.sqrt((lidar[0][0]-aruco[0][0]) * (lidar[0][0]-aruco[0][0]) + (lidar[0][1]-aruco[0][1]) * (lidar[0][1]-aruco[0][1]))
    d_bbox = np.sqrt((lidar[0][0]-bbox[0][0]) * (lidar[0][0]-bbox[0][0]) + (lidar[0][1]-bbox[0][1]) * (lidar[0][1]-bbox[0][1]))
    
    #convert distances from pixels to metres
    dist_aruco = d_aruco * markerLength / ((msp4+msp)/2)
    dist_bbox = d_bbox * markerLength / ((msp4+msp)/2)
    
    return dist_aruco, dist_bbox




height, width = 2160, 3840 #fixed input image/video resolution
markerLengthOrg = 0.55 #real size of the marker in metres, this value does not change in algorithm
markerLength = markerLengthOrg #real size of the marker in metres, this value changes in algorithm
marker_div = 1.2 #correction for altitude estimation from marker
div = 1.013 #additional correction for distance calculation (based on altitude test)
DIFF_MAX = 2/3 * step_frame * 2 #maximum displacement of ArUco centre between frames with vehicle speed of 72 km/h = 20 m/s

if useCentroidData:
    centroid_data = readCentroidData(path_dcnn_data) #read centroid data from DCNN
if saveResults:
    file = outputDataInit() #initialize output file for saving results


#MARKER DETECTION AND POINTS CALCULATIONS

    #if any marker was detected
    if np.all(ids != None):
        #estimate pose of detected markers
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)
