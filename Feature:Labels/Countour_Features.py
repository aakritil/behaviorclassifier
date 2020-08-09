#!/usr/bin/env python
# coding: utf-8

# In[72]:


import argparse
import sys
import csv
import random
import cv2
import shapely.geometry as geometry
import math
import sympy as sym
import scipy.misc
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.spatial.distance import cdist
import matplotlib.gridspec as gridspec

# globals

frames = {}  # frames object - holds all framedata

# fame processing variable
frameData=[]
togetherDataRandC = 0
togetherDataLandC = 0

def getLongestPt(points, x, y):
    points.sorted(key = lambda p: sqrt((p.x - x)**2 + (p.y - y)**2))
    return points
    
def getMeanandStdev(myarray, behavior):
    # convert to NP array
    NPArray=np.array(myarray)  
    return ([behavior, np.mean(NPArray), np.std(NPArray)]) 
    
def calcDistxy (pt1,pt2):
    if len(pt1) < 2 or len(pt2) < 2:
        print("points not correct")
    else:
        return math.sqrt((pt2[0]-pt1[0])**2+(pt2[1]-pt1[1])**2)

def calcovingAvg(mylist, N):
    cumsum, moving_aves = [0], []
    for i in range(0,N-1):
        moving_aves.append(0)
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)       
    return moving_aves

def calcAnglew3(c1, c2, c3):
    ang = 0.0
    ang = math.degrees(math.atan2(c3.cy-c2.cy, c3.cx-c2.cx) - math.atan2(c1.cy-c2.cy, c1.cx-c2.cx))
    if ang < 0 :
        return ang*(-1)
    else:
        return ang
    
def calcAnglewg3(array, array2, array3):
    ang = 0.0
    ang = math.degrees(math.atan2(array3[1]-array2[1], array3[0]-array2[0]) - math.atan2(array[1]-array2[1], array[1]-array2[0]))
    if ang < 0 :
        return ang+360
    else:
        return ang



#corners
OuterUpperRight=[]
OuterUpperLeft=[]
OuterLowerRight=[]
OuterLowerLeft=[]
InnerUpperRight=[]
InnerUpperLeft=[]
InnerLowerRight=[]
InnerLowerLeft=[]

class Frame:
    frame_num = 0
    contours = []
    ctorDist = 0.0
    ctolDist = 0.0 
    voleRPositionInCountor = -1
    voleCPositionInCountor = -1
    voleLPositionInCountor = -1

    def __init__(self, frame_num, contours):
        self.frame_num = frame_num
        self.contours = contours
        #included in init to reduce looping 07/17
        #self.voleDist()
        
        contourCount=0
        #iterate though the contours 
        for contourCount in range(len(self.contours)):
            # check the lable 
            currentLable = self.contours[contourCount].labels
            for lCount in range(len(currentLable)):
                #iterate through contour assumes if the count is 3 they all seperate contours
                # set the Vole position 
                if ( currentLable[lCount] == 'vole_r'):
                    self.voleRPositionInCountor = contourCount
                elif ( currentLable[lCount] == 'vole_c'):
                    self.voleCPositionInCountor = contourCount
                else:
                    self.voleLPositionInCountor = contourCount
                #find the distance

        # calc ctor and ctol calc based on contour position to save cpu
        if self.voleCPositionInCountor == self.voleRPositionInCountor :
            # set distance and together flag
            self.ctorDist = 0.0
            self.ctolDist = self.contours[self.voleCPositionInCountor].calcDistanceFromContour(self.contours[self.voleLPositionInCountor])
            
        elif self.voleCPositionInCountor == self.voleLPositionInCountor :
            # set distance and together flag
            self.ctolDist = 0.0
            self.ctorDist = self.contours[self.voleCPositionInCountor].calcDistanceFromContour(self.contours[self.voleRPositionInCountor])
        else:          
            self.ctorDist = self.contours[self.voleCPositionInCountor].calcDistanceFromContour(self.contours[self.voleRPositionInCountor])
            self.ctolDist = self.contours[self.voleCPositionInCountor].calcDistanceFromContour(self.contours[self.voleLPositionInCountor])
            
        # calc from previour contour if not first frame
        if frame_num != 0:
            # get the list contours from the last frame could be multiple
            oldContours = frames[frame_num-1].getContours()
            oldCposition=frames[frame_num-1].voleCPositionInCountor
            oldRposition=frames[frame_num-1].voleRPositionInCountor
            oldLposition=frames[frame_num-1].voleLPositionInCountor

            # call C regardless as we need it in any case 
            self.contours[self.voleCPositionInCountor].calcVelocityandAcceleration(oldContours[oldCposition], 1, 30)
            self.contours[self.voleCPositionInCountor].calcNosept(oldContours[oldCposition])
            self.contours[self.voleCPositionInCountor].calcAngleofOrientation(oldContours[oldCposition])
            self.contours[self.voleCPositionInCountor].calcEccentricity(oldContours[oldCposition])
            self.contours[self.voleCPositionInCountor].calcAreaOverlap(oldContours[oldCposition])


            # if c to r together then only call L 
            if self.ctorDist == 0.0:
                self.contours[self.voleLPositionInCountor].calcVelocityandAcceleration(oldContours[oldLposition], 1, 30)
                self.contours[self.voleLPositionInCountor].calcNosept(oldContours[oldLposition])
                self.contours[self.voleLPositionInCountor].calcAngleofOrientation(oldContours[oldLposition])
                self.contours[self.voleLPositionInCountor].calcEccentricity(oldContours[oldLposition])
                self.contours[self.voleLPositionInCountor].calcAreaOverlap(oldContours[oldLposition])

            # if c to l together then only call R
            elif self.ctolDist == 0.0:
                self.contours[self.voleRPositionInCountor].calcVelocityandAcceleration(oldContours[oldRposition], 1, 30)
                self.contours[self.voleRPositionInCountor].calcNosept(oldContours[oldRposition])
                self.contours[self.voleRPositionInCountor].calcAngleofOrientation(oldContours[oldRposition])
                self.contours[self.voleRPositionInCountor].calcEccentricity(oldContours[oldRposition])
                self.contours[self.voleRPositionInCountor].calcAreaOverlap(oldContours[oldRposition])   
        # they are seperate call both L and R
            else:
                self.contours[self.voleLPositionInCountor].calcVelocityandAcceleration(oldContours[oldLposition], 1, 30)
                self.contours[self.voleLPositionInCountor].calcNosept(oldContours[oldLposition])
                self.contours[self.voleLPositionInCountor].calcAngleofOrientation(oldContours[oldLposition])
                self.contours[self.voleLPositionInCountor].calcEccentricity(oldContours[oldLposition])
                self.contours[self.voleLPositionInCountor].calcAreaOverlap(oldContours[oldLposition])
                                                                                              
                self.contours[self.voleRPositionInCountor].calcVelocityandAcceleration(oldContours[oldRposition], 1, 30)
                self.contours[self.voleRPositionInCountor].calcNosept(oldContours[oldRposition])
                self.contours[self.voleRPositionInCountor].calcAngleofOrientation(oldContours[oldRposition])
                self.contours[self.voleRPositionInCountor].calcEccentricity(oldContours[oldRposition])
                self.contours[self.voleRPositionInCountor].calcAreaOverlap(oldContours[oldRposition]) 
        if frame_num > 20:
            oldContours2 = frames[frame_num-20].getContours()
            oldCposition2=frames[frame_num-20].voleCPositionInCountor
            oldRposition2=frames[frame_num-20].voleRPositionInCountor
            oldLposition2=frames[frame_num-20].voleLPositionInCountor
            
            self.contours[self.voleCPositionInCountor].calcAreaOverlapframe(oldContours2[oldCposition2])
                

    def getContours(self):
        return self.contours
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"[Frame: {self.frame_num}, Contours: {self.contours}, ctorDist: {self.ctorDist}, ctolDist: {self.ctolDist}]"                     
    
    def calcDistance(self, fromCx, fromCy):
        dist= math.sqrt((fromCx-self.cx)**2+(fromCy-self.cy)**2)
        return dist
    
    def getAreaOverlap(self):
        return[self.contours[self.voleCPositionInCountor].areaoverlapmultiframe]

    
    def getDataArray(self):
        return [self.frame_num, 
                self.ctorDist, 
                self.ctolDist, 
                self.voleRPositionInCountor, 
                self.voleCPositionInCountor, 
                self.voleLPositionInCountor, 
                self.contours[self.voleCPositionInCountor].velocity, 
                self.contours[self.voleRPositionInCountor].velocity,
                self.contours[self.voleLPositionInCountor].velocity,
                self.contours[self.voleCPositionInCountor].acceleration, 
                self.contours[self.voleRPositionInCountor].acceleration,
                self.contours[self.voleLPositionInCountor].acceleration,
                self.contours[self.voleCPositionInCountor].angofOrientation, 
                self.contours[self.voleRPositionInCountor].angofOrientation,
                self.contours[self.voleLPositionInCountor].angofOrientation,  
                self.contours[self.voleCPositionInCountor].rawangofOrientation,
                self.contours[self.voleRPositionInCountor].rawangofOrientation,
                self.contours[self.voleLPositionInCountor].rawangofOrientation,
                self.contours[self.voleCPositionInCountor].longestDist, 
                self.contours[self.voleRPositionInCountor].longestDist,
                self.contours[self.voleLPositionInCountor].longestDist, 
                #self.contours[self.voleCPositionInCountor].circularity, 
                #self.contours[self.voleRPositionInCountor].circularity,
                #self.contours[self.voleLPositionInCountor].circularity, 
                self.contours[self.voleCPositionInCountor].avgDistanceFromCentroidToPoints, 
                self.contours[self.voleRPositionInCountor].avgDistanceFromCentroidToPoints,
                self.contours[self.voleLPositionInCountor].avgDistanceFromCentroidToPoints, 
                self.contours[self.voleCPositionInCountor].cummulativeDist, 
                self.contours[self.voleRPositionInCountor].cummulativeDist,
                self.contours[self.voleLPositionInCountor].cummulativeDist, 
                self.contours[self.voleCPositionInCountor].area, 
                self.contours[self.voleRPositionInCountor].area,
                self.contours[self.voleLPositionInCountor].area, 
                #this must be removed, data has to be in a form of number, not point 
                #self.contours[self.voleCPositionInCountor].cx,
                #self.contours[self.voleCPositionInCountor].cy,
                #self.contours[self.voleRPositionInCountor].cx,
                #self.contours[self.voleRPositionInCountor].cy,
                #self.contours[self.voleLPositionInCountor].cx,
                #self.contours[self.voleLPositionInCountor].cy,
                self.contours[self.voleCPositionInCountor].x_v1,
                self.contours[self.voleCPositionInCountor].y_v1,
                self.contours[self.voleCPositionInCountor].majoraxislen,
                self.contours[self.voleCPositionInCountor].minoraxislen,
                self.contours[self.voleRPositionInCountor].majoraxislen,
                self.contours[self.voleRPositionInCountor].minoraxislen,
                self.contours[self.voleLPositionInCountor].majoraxislen,
                self.contours[self.voleLPositionInCountor].minoraxislen,
                self.contours[self.voleCPositionInCountor].chamber,
                self.contours[self.voleCPositionInCountor].distOUR, 
                self.contours[self.voleRPositionInCountor].distOUR,
                self.contours[self.voleLPositionInCountor].distOUR,
                self.contours[self.voleCPositionInCountor].distOUL, 
                self.contours[self.voleRPositionInCountor].distOUL,
                self.contours[self.voleLPositionInCountor].distOUL,
                self.contours[self.voleCPositionInCountor].distOLR,
                self.contours[self.voleRPositionInCountor].distOLR,
                self.contours[self.voleLPositionInCountor].distOLR,
                self.contours[self.voleCPositionInCountor].distOLL, 
                self.contours[self.voleRPositionInCountor].distOLL,
                self.contours[self.voleLPositionInCountor].distOLL,
                self.contours[self.voleCPositionInCountor].distIUR, 
                self.contours[self.voleRPositionInCountor].distIUR,
                self.contours[self.voleLPositionInCountor].distIUR,
                self.contours[self.voleCPositionInCountor].distIUL, 
                self.contours[self.voleRPositionInCountor].distIUL,
                self.contours[self.voleLPositionInCountor].distIUL,
                self.contours[self.voleCPositionInCountor].distILR, 
                self.contours[self.voleRPositionInCountor].distILR,
                self.contours[self.voleLPositionInCountor].distILR,
                self.contours[self.voleCPositionInCountor].distILL, 
                self.contours[self.voleRPositionInCountor].distILL,
                self.contours[self.voleLPositionInCountor].distILL,
                self.contours[self.voleCPositionInCountor].rotationalVel,
                self.contours[self.voleRPositionInCountor].rotationalVel,
                self.contours[self.voleLPositionInCountor].rotationalVel,
                self.contours[self.voleCPositionInCountor].eccentricity2,
                self.contours[self.voleRPositionInCountor].eccentricity2,
                self.contours[self.voleLPositionInCountor].eccentricity2,
                self.contours[self.voleCPositionInCountor].eccentricity1,
                self.contours[self.voleRPositionInCountor].eccentricity1,
                self.contours[self.voleLPositionInCountor].eccentricity1,
                self.contours[self.voleCPositionInCountor].eccentricitywdist,
                self.contours[self.voleRPositionInCountor].eccentricitywdist,
                self.contours[self.voleLPositionInCountor].eccentricitywdist,
                self.contours[self.voleCPositionInCountor].closetocentroid,
                self.contours[self.voleRPositionInCountor].closetocentroid,
                self.contours[self.voleLPositionInCountor].closetocentroid,
                self.contours[self.voleCPositionInCountor].longPtSlope,
                self.contours[self.voleRPositionInCountor].longPtSlope,
                self.contours[self.voleLPositionInCountor].longPtSlope,
                self.contours[self.voleCPositionInCountor].togetherFlag,
                self.contours[self.voleRPositionInCountor].togetherFlag,
                self.contours[self.voleLPositionInCountor].togetherFlag,
                self.contours[self.voleCPositionInCountor].distFromPreviousCont,
                self.contours[self.voleRPositionInCountor].distFromPreviousCont,
                self.contours[self.voleLPositionInCountor].distFromPreviousCont,
                self.contours[self.voleCPositionInCountor].angleOfMovement,
                self.contours[self.voleRPositionInCountor].angleOfMovement,
                self.contours[self.voleLPositionInCountor].angleOfMovement,
                self.contours[self.voleCPositionInCountor].areaoverlap,
                self.contours[self.voleRPositionInCountor].areaoverlap,
                self.contours[self.voleLPositionInCountor].areaoverlap             
                
               ]

    
    

class Contour:
    trace_id = 0
    labels = []
    points = []
    cx = 0.0
    cy = 0.0
    velocity = 0.0
    acceleration = 0.0
    longestPt = []
    avgDistanceFromCentroidToPoints = 0.0
    area = 0.0
    longestDist = 0.0
    longPtSlope = 0.0
    tortuisity = 0.0
    circularity = 0.0
    togetherFlag = 0
    x_v1 = 0.0
    y_v1 = 0.0
    x_v2 = 0.0
    y_v2 = 0.0  
    angofOrientation = 0.0
    rawangofOrientation = 0.0 
    ellipse = []
    cummulativeDist = 0.0 
    dx= 0.0
    dy = 0.0
    d2x =0.0
    d2y = 0.0
    eccentricity1 = 0.0
    eccentricity2=0.0
    majorPt = []
    minorPt = []
    angleOfMovement = 0.0
    chamber=0.0
    distOUR = 0.0
    distOUL = 0.0
    distOLR = 0.0
    distOLL = 0.0
    distIUR = 0.0
    distIUL = 0.0
    distILR = 0.0
    distILL = 0.0
    rotationalVel=0.0
    maybenose=[]
    maybenose2=[]
    nose=[]
    closetocentroid=0
    minoraxislen=0.0
    majoraxislen= 0.0
    distFromPreviousCont=0.0
    areaoverlap=0.0
    eccentricitywdist = 0.0
    areaoverlapmultiframe = 0.0
    

    
    def __init__(self, trace_id, labels, points):
        self.trace_id = trace_id
        self.labels = labels
        self.points = points
        
        #fit ellipse
        self.ellipse = cv2.fitEllipse(points)
        
        #set together flag if there are more than one lable 
        if len(labels) == 2:
            self.togetherFlag = 1  
        
        #calculate cx and cy
        moments = cv2.moments(self.points)
        self.cx = moments['m10']/moments['m00']
        self.cy = moments['m01']/moments['m00']
        
        #Calculate location
        self.calcLocation(OuterUpperRight, OuterUpperLeft, OuterLowerRight, OuterLowerLeft, InnerUpperRight, InnerUpperLeft,InnerLowerRight,InnerLowerLeft)
        
        #first calculate the longest point using current contour
        self.calcLongestPointforSelf()
        
        #calc major and minor axes
        self.majorMinorAxis()
        
        #calc centroid dist 5 frames ago
        self.calcCentroidMinordist(5)
        
        # Find Chamber num
        if self.togetherFlag == 1:            
            for i in range(len(labels)):
                if labels[i] == 'vole_l':
                    self.chamber = 1
                elif labels[i] == 'vole_r':
                    self.chamber = 3
        else: 
            self.calcChamber(OuterUpperRight, OuterUpperLeft, OuterLowerRight, OuterLowerLeft, InnerUpperRight, InnerUpperLeft,InnerLowerRight,InnerLowerLeft)       

    def setLabels(self, labels):
        self.labels = labels
        
    def getLabels(self):
        return self.labels

    def getPoints(self):
        return self.points

    def __repr__(self):
        return self.__str__()
    
    def getCx(self):
        return self.cx
    
    def getCy(self):
        return self.cy

    def __str__(self):
        return f"[contour: Labels: {self.labels}, angofOrientation: {self.angofOrientation}, avgDistanceFromCentroidToPoints: {self.avgDistanceFromCentroidToPoints},  cummulativeDist: {self.cummulativeDist}, longestPt: {self.longestPt}, velocity: {self.velocity},circularity: {self.circularity}, CX: {self.cx}, CY: {self.cy} ]"
    
    def calcDistance(self, fromCx, fromCy):
        return math.sqrt((fromCx-self.cx)**2+(fromCy-self.cy)**2)

    def calcAngle(self, fromCx, fromCy):
        angle = math.degrees(np.arctan2(fromCy-self.cy,fromCx-self.cx))
        if angle < 0:
            angle= angle+360
        return angle

    def calcDistanceFromContour(self, fromContour):
        dist= math.sqrt((fromContour.cx-self.cx)**2+(fromContour.cy-self.cy)**2)
        return dist
        
    def calcVelocityandAcceleration( self, OldCountour, frameLength,frameRate):
            ocx = OldCountour.getCx()
            ocy = OldCountour.getCy()
            self.dx= self.cx - ocx
            self.dy= self.cy - ocy
            self.d2x= self.dx - OldCountour.dx
            self.d2y= self.dy - OldCountour.dy
            self.distFromPreviousCont = self.calcDistance(ocx,ocy)
            self.angleOfMovement = self.calcAngle(ocx,ocy)
            self.velocity= math.sqrt(self.dx**2+self.dy**2)/(frameLength*(frameRate/60))
            self.acceleration = math.sqrt(self.d2x**2+self.d2y**2)/(frameLength*(frameRate/60))
            self.cummulativeDist = OldCountour.cummulativeDist + self.distFromPreviousCont
            self.rotationalVel= abs((int(self.angleOfMovement) - int(OldCountour.angleOfMovement))/(frameLength*(frameRate/60)))
            
    def calcLongestPointforSelf(self):
        distance = 0.0
        averageDist = 0.0
        alldist = []
        #iterate through the points
        for pts in range(len(self.points)):
            #calculate distance between centriod and point
            dist = self.calcDistance((self.points[pts])[0],(self.points[pts])[1])
            alldist.append(dist)
            if dist > distance :
                # assign if the current point is longest
                distance = dist
                longestPt = self.points[pts]
                averageDist = averageDist+dist
            else:
                averageDist = averageDist+dist
       
        alldistnp = np.array(alldist)
        # assign the values   
        self.eccentricitywdist = np.std(alldistnp)
        self.longestPt = longestPt
        self.avgDistanceFromCentroidToPoints = averageDist / len(self.points)
        self.longestDist = distance
        self.getLPSlope()
        
    def calcCircularity(self):
        hypotheticalRadius = self.area / (2*math.pi)
        self.circularity = (self.avgDistanceFromCentroidToPoints)/(hypotheticalRadius)
        #when this value is closer to 1, the area is more circular.
    
    def calcEccentricity(self, oldContour):
        if len(self.majorPt) == 2 and len(self.minorPt) == 2:
            self.eccentricity1 = abs(1-(self.ellipse[1][1]/self.ellipse[1][0]))
            self.majoraxislen = math.sqrt((self.majorPt[0][0]-self.majorPt[1][0])**2+(self.majorPt[0][1]-self.majorPt[1][1])**2)
            self.minoraxislen = math.sqrt((self.minorPt[0][0]-self.minorPt[1][0])**2+(self.minorPt[0][1]-self.minorPt[1][1])**2)
            eccentricity=  self.majoraxislen/self.minoraxislen
            self.eccentricity2 = abs(1-eccentricity)
        else:
            self.eccentricity1 = oldContour.eccentricity1
            self.majoraxislen = oldContour.majoraxislen
            self.minoraxislen = oldContour.minoraxislen
            self.eccentricity2 = oldContour.eccentricity2 
            
        #when this value is closer to 1, the area is less eccentric
        
    def majorMinorAxis(self):
        
        pxarray=[]
        pyarray=[]
        npx=[]
        npy=[]
        topcurve=[]
        bottomcurve=[]
        
        for pts in range(len(self.points)):    
            #iterate through the points get px and py array
            # get the x coordinates and the y coordinates 
            pxarray.append(self.points[pts][0])
            pyarray.append(self.points[pts][1]) 

        # calc mean of x and y
        npxMean = np.mean(pxarray)
        npyMean = np.mean(pyarray)
        
        #subrtract mean from the x and y
        for pts in range(len(pxarray)):
            # get the x coordinates and the y coordinates 
            npx.append( pxarray[pts] - npxMean ) 
            npy.append( pyarray[pts] - npyMean ) 
            
        # make a matrix with the normalized values
        meanMatrix = np.vstack([npx,npy])
        actualMatrix=np.vstack([pxarray,pyarray])

        #from this matrix, calculated the covariance and the eigenvectors 
        covarianceM = np.cov(meanMatrix)
        eigenVal, eigenVector= np.linalg.eig(covarianceM)
        
        #figure out which one is the longest and which one is smaller
        sortedVector = np.argsort(eigenVal)[::-1]
        self.x_v1, self.y_v1 = eigenVector[:, sortedVector[0]]  # larger-major
        self.x_v2, self.y_v2 = eigenVector[:, sortedVector[1]]
        
        self.majorPt = self.calcEndPoint(self.x_v1, self.y_v1, pxarray, pyarray) #major
        
        self.minorPt = self.calcEndPoint(self.x_v2, self.y_v2, pxarray, pyarray) #minor
        

    def calcCurvature(self , arrayofcoords):
        dx_dt = np.gradient(arrayofcoords[:, 0])
        dy_dt = np.gradient(arrayofcoords[:, 1])
        velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
        ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
        tangent = np.array([1/ds_dt] * 2).transpose() * velocity
        tangent_x = tangent[:, 0]
        tangent_y = tangent[:, 1]
        deriv_tangent_x = np.gradient(tangent_x)
        deriv_tangent_y = np.gradient(tangent_y)
        dT_dt = np.array([ [deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])
        length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)
        normal = np.array([1/length_dT_dt] * 2).transpose() * dT_dt
        d2s_dt2 = np.gradient(ds_dt)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
        return curvature  
    
    def calcAreaOverlap(self,oldcontour):
        
        contourpoly=geometry.Polygon(self.points)
        oldcontourpoly = geometry.Polygon(oldcontour.points)
        contourpolybuffered=contourpoly.buffer(0)
        oldcontourpolybuffered = oldcontourpoly.buffer(0)
        
        if contourpolybuffered.geom_type == 'MultiPolygon' or contourpolybuffered.geom_type == 'Polygon' and oldcontourpolybuffered.geom_type == 'MultiPolygon' or oldcontourpolybuffered.geom_type == 'Polygon':
            areaoverlap = contourpolybuffered.intersection(oldcontourpolybuffered).area
        else:
            areaoverlap = 0
        self.areaoverlap = areaoverlap/self.area
    
    def calcAreaOverlapframe(self,oldcontour):
        
        contourpoly=geometry.Polygon(self.points)
        oldcontourpoly = geometry.Polygon(oldcontour.points)
        contourpolybuffered=contourpoly.buffer(0)
        oldcontourpolybuffered = oldcontourpoly.buffer(0)
        
        if contourpolybuffered.geom_type == 'MultiPolygon' or contourpolybuffered.geom_type == 'Polygon' and oldcontourpolybuffered.geom_type == 'MultiPolygon' or oldcontourpolybuffered.geom_type == 'Polygon':
            areaoverlap = contourpolybuffered.intersection(oldcontourpolybuffered).area
        else:
            areaoverlap = 0
            
        print(areaoverlap)
        print(contourpolybuffered.geom_type)
        print(oldcontourpolybuffered.geom_type)
        
        self.areaoverlapmultiframe = areaoverlap/self.area
        
    def calcNosept (self,oldcontour):
        if len(self.majorPt) == 2 :
            dist1=calcDistxy([self.longestPt[0],self.longestPt[1]],self.majorPt[0])
            dist2=calcDistxy([self.longestPt[0],self.longestPt[1]],self.majorPt[1])
            if dist1 > dist2:
                self.maybenose = self.majorPt[0]
                self.maybenose2=self.majorPt[1]
            else:
                self.maybenose = self.majorPt[1]
                self.maybenose2=self.majorPt[0]

            if oldcontour.togetherFlag==1 and self.togetherFlag ==0:
                self.nose= self.maybenose

            elif len(oldcontour.maybenose) ==2 and calcDistxy(self.maybenose,oldcontour.nose) > 20:
                self.nose = self.maybenose2
            else:
                self.nose=self.maybenose 
        else:
            self.nose=oldcontour.nose
            self.maybenose2=oldcontour.maybenose2
            self.maybenose =oldcontour.maybenose 
            
        
    def calcEndPoint(self, xv, yv, px, py):       
        #figure out the endpoints of each axis
        MApts = [(self.cx-1000*(xv),self.cy-1000*(yv)), (self.cx+1001*(xv),self.cy+1001*(yv))]
        MAline = geometry.LineString(MApts)

        contourpoly = geometry.Polygon(zip(px,py))
        contourpolybuffered = contourpoly.buffer(0)
        px.append(px[0])
        py.append(py[0])
        
        if contourpoly.is_valid :
            MAendpts = contourpoly.exterior.intersection(MAline) 
        elif contourpolybuffered.geom_type == 'MultiPolygon' or contourpolybuffered.centroid.is_empty:
            contourlinestring=geometry.LineString(zip(px,py))
            MAendpts = contourlinestring.intersection(MAline) 
        else:
            MAendpts = contourpolybuffered.exterior.intersection(MAline)
    
        
        if MAendpts.is_empty:
            print("shapes don't intersect")
    
        MAendptsarray=np.array(MAendpts)
        
        endpoints=[]
        magnitude=0.0
        mag1=-1
        mag2 =-1
        pt1pos = 0
        pt2pos = 0
        
        #Point(0,0).distance(Point(1,1))
        check = 0 
        
        # get top two points 
        if len(MAendptsarray) > 2:      
            check =1
            for i in range(len(MAendptsarray)):
                magnitudet= self.calcDistance( MAendptsarray[i][0],MAendptsarray[i][1])
                if magnitudet > mag1 :
                    mag2 = mag1
                    pt2pos = pt1pos
                    mag1 = magnitudet
                    pt1pos = i
                elif magnitudet > mag2 :
                    mag2 = magnitudet
                    pt2pos = i
                    
            endpoints.append([MAendptsarray[pt1pos][0]] + [MAendptsarray[pt1pos][1]] )
            endpoints.append([MAendptsarray[pt2pos][0]] + [MAendptsarray[pt2pos][1] ])
        elif len(MAendptsarray) == 0: 
            check = 2
            print("cx=%s, cy=%s"%(self.cx,  self.cy))
            print("xv=%s, yv=%s"%(xv,  yv))
            print(contourpoly.centroid)
            print(contourpolybuffered.centroid)
            # problem when only one point is returned 
        elif  len(MAendptsarray) == 2 and MAendpts.geom_type == 'MultiPoint'  :
            check = 3
            endpoints.append([MAendptsarray[0][0]] + [MAendptsarray[0][1]] )
            endpoints.append([MAendptsarray[1][0]] + [MAendptsarray[1][1]])
        elif MAendpts.geom_type == 'Point':
            check = 4
            endpoints.append([MAendptsarray[0]] + [MAendptsarray[1]] )

        return endpoints
    
    def calcCentroidMinordist (self, dist):
        if len(self.minorPt) == 2:
            dist1 = calcDistxy(self.minorPt[0],[self.cx,self.cy])
            dist2 = calcDistxy(self.minorPt[1],[self.cx,self.cy])
            if dist1 <= dist or dist2 <= dist:
                self.closetocentroid = 1
            else:
                self.closetocentroid = 0
        elif len(self.minorPt) == 1:
            dist1 = calcDistxy(self.minorPt[0],[self.cx,self.cy])
            if dist1 <= dist:
                self.closetocentroid = 1
            else:
                self.closetocentroid = 0
            
    def calcAngleofOrientation(self, oldcontour):

        if len(self.majorPt) == 2 :
            ao =0.0
            rao = 0.0
            ao = math.degrees(np.arctan2(self.majorPt[0][0]-self.cx, self.majorPt[0][1]-self.cy))
            rao = ao

            if ao < 0:
                ao= ao+360

            if 150 < abs(ao - oldcontour.angofOrientation) < 200 :
                ao = math.degrees(np.arctan2(self.majorPt[1][0]-self.cx, self.majorPt[1][1]-self.cy))
                rao = ao
                if ao < 0:
                    ao = ao+360

            self.angofOrientation = ao
            if ao > 180:
                self.rawangofOrientation = -(360-ao)
            else:
                self.rawangofOrientation = self.angofOrientation
        else:
            self.rawangofOrientation = oldcontour.rawangofOrientation
            self.angofOrientation = oldcontour.angofOrientation
        
    
    def calcChamber(self, OuterUpperRight, OuterUpperLeft, OuterLowerRight, OuterLowerLeft, InnerUpperRight, InnerUpperLeft,InnerLowerRight,InnerLowerLeft):
        #what chamber?
        centroidpt=geometry.Point(self.cx,self.cy)
        
        chamber1=geometry.Polygon([(OuterLowerRight[0],OuterLowerRight[1]),
                                   (OuterUpperRight[0],OuterUpperRight[1]),
                                   (InnerUpperRight[0],InnerUpperRight[1]),
                                   (InnerLowerRight[0],InnerLowerRight[1])])
        if chamber1.contains(centroidpt) == True:
            self.chamber=1
            return self.chamber
       
            
        chamber2=geometry.Polygon([(InnerLowerRight[0],InnerLowerRight[1]),
                                   (InnerUpperRight[0],InnerUpperRight[1]),
                                   (InnerUpperLeft[0],InnerUpperLeft[1]),
                                   (InnerLowerLeft[0],InnerLowerLeft[1])])
        if chamber2.contains(centroidpt) == True:
            self.chamber=2
            return self.chamber
        
        chamber3=geometry.Polygon([(InnerLowerLeft[0],InnerLowerLeft[1]),
                                   (InnerUpperLeft[0],InnerUpperLeft[1]),
                                   (OuterUpperLeft[0],OuterUpperLeft[1]),
                                   (OuterLowerLeft[0],OuterLowerLeft[1])])
        
        if chamber3.contains(centroidpt) == True:
            self.chamber=3
            return self.chamber
               
        
    def calcLocation(self, OuterUpperRight, OuterUpperLeft, OuterLowerRight, OuterLowerLeft, InnerUpperRight, InnerUpperLeft,InnerLowerRight,InnerLowerLeft):
        #distance from the outer corners 
        self.distOUR = self.calcDistance(OuterUpperRight[0],OuterUpperRight[1])
        self.distOUL = self.calcDistance(OuterUpperLeft[0],OuterUpperLeft[1])
        self.distOLR = self.calcDistance(OuterLowerRight[0],OuterLowerRight[1])
        self.distOLL = self.calcDistance(OuterLowerLeft[0],OuterLowerLeft[1])
        #distance from the inner corners 
        self.distIUR = self.calcDistance(InnerUpperRight[0],InnerUpperRight[1])
        self.distIUL = self.calcDistance(InnerUpperLeft[0],InnerUpperLeft[1])
        self.distILR = self.calcDistance(InnerLowerRight[0],InnerLowerRight[1])
        self.distILL = self.calcDistance(InnerLowerLeft[0],InnerLowerLeft[1])
        
        
    def getLPSlope (self):
        n=0
        b = (self.cx-self.longestPt[0])
        d = (self.cy-self.longestPt[1])
        if b != 0:
            n = (d)/(b)
        self.longPtSlope=n


# In[73]:


import time 

def readCountorFile(rcontours_file,  last_frame):
    
    file = open(rcontours_file, "r")
    try:
        contour_reader = csv.reader(file)
        for row in contour_reader:
            frame_num = int(row[0])

            if (last_frame + 1 < frame_num):
                # gap frames
                print(f"Backfilling gap frame {last_frame+1}")
                while (last_frame + 1 < frame_num):
                    frames[last_frame + 1] = Frame(last_frame + 1, [])
                    last_frame = last_frame + 1
            i = 1
            contours = []
            # BEGIN WHILE LOOP TO GET CONTOURS 
            while (i+2 < len(row)):
                contour_trace_id = row[i]
                labels = []
                label = row[i+1]
                # get and split the label
                if len(label) > 0:
                    labels = label.split("+")

                # get the point data
                pointdata = [int(item) for item in row[i+2].split(",")]
                pointdata = np.reshape(pointdata, (-1, 2))

                # ADDING TO CHECK IF FUNCTION WORKS
                area = cv2.contourArea(pointdata)
                if area == 0 or area < 1:
                    print("Area ({area}) of one of these contours is < 1 frame: {frame_num}")

                # add countrous to the coutors list 
                # create countour object 
                cntr= Contour(contour_trace_id, labels, pointdata)
                # set the area to save CPU
                cntr.area = area
                contours.append(cntr)

                # increase the loop coufnonnt 
                i = i + 3
            # END WHILE LOOP TO GET CONTOURS 
            # add frames to frame array
            frames[frame_num] = Frame(frame_num, contours)

            # set the last frame number
            last_frame = frame_num

            #dummy code to exit loop after frame 0
            #if frame_num == 10:
                #print ("breaking up!")
                #break
        print ("finished building frames for  %s at %s"%(rcontours_file, time.asctime()))
    finally:
        file.close()
    
print ("starting code %s"%time.asctime())
output_filepath = "//Users//aakritilakshmanan//Desktop//vole_contour_data//outputnewdataxx.aakriti.csv" 
framebad = "/Users/aakritilakshmanan/Downloads/frame_bad.csv"
bottominputfile = "//Users//aakritilakshmanan//Desktop//vole_contour_data//manual_20191108161215.bottom.resolved.contours"
topinputfile = "//Users//aakritilakshmanan//Desktop//vole_contour_data//manual_20191108161215.top.resolved.contours.csv"
bottominputfile2 = "//Users//aakritilakshmanan//Desktop//vole_contour_data//manual_20191108161442.bottom.resolved.contours"
topinputfile2 = "//Users//aakritilakshmanan//Desktop//vole_contour_data//manual_20191108161442.top.resolved.contours"


manual_20191108165756_bottom= "//Users//aakritilakshmanan//Downloads//manual_20191108165756.bottom.resolved.contours"
manual_20191108165756_top="//Users//aakritilakshmanan//Downloads//manual_20191108165756.top.resolved.contours"
manual_20191108170203_bottom= "//Users//aakritilakshmanan//Downloads//manual_20191108170203.bottom.resolved.contours"
manual_20191108170203_top="//Users//aakritilakshmanan//Downloads//manual_20191108170203.top.resolved.contours"
manual_20191108143820_bottom= "//Users//aakritilakshmanan//Downloads//manual_20191108143820.bottom.resolved.contours"
manual_20191108143820_top= "//Users//aakritilakshmanan//Downloads//manual_20191108143820.top.resolved.contours"
manual_20191108152515_top= "//Users//aakritilakshmanan//Downloads//manual_20191108152515.top.resolved.contours"
manual_20191108152703_bottom= "//Users//aakritilakshmanan//Downloads//manual_20191108152703.bottom.resolved.contours"


# In[2]:


def ProcessFrameData(file,our, oul, olr, oll , iur, iul, ilr, ill):
    global frameData
    global togetherDataRandC 
    global togetherDataLandC
    
    #corners bottom
    global OuterUpperRight 
    global OuterUpperLeft 
    global OuterLowerRight 
    global OuterLowerLeft 
    global InnerUpperRight 
    global InnerUpperLeft 
    global InnerLowerRight 
    global InnerLowerLeft 

    OuterUpperRight= our
    OuterUpperLeft=oul
    OuterLowerRight=olr
    OuterLowerLeft=oll
    InnerUpperRight=iur
    InnerUpperLeft=iul
    InnerLowerRight=ilr
    InnerLowerLeft=ill

    # read countor add to global frames array 
    frmcount = readCountorFile(file, -1)
    
    # FOR LOOP BEGIN
    for frameCount in range( len(frames)-1):
        # capture time spend Vole R and L based on frame count    
        if (frames[frameCount].ctorDist) != 0 :
            togetherDataRandC = togetherDataRandC + 1
        elif (frames[frameCount].ctolDist) != 0 :
            togetherDataLandC = togetherDataLandC + 1
        frameData.append(frames[frameCount].getDataArray())
    # FOR LOOP END
# end of process frame data

frameData=[]

#GENERATING FEATURES

#ProcessFrameData(bottominputfile,[1066,302],[376,288], [1061,544], [367,522], [828,297],[596,295],[827,541], [595,527]  )
#ProcessFrameData(topinputfile,[1072,54],[385,52],[1065,290],[376,277],[836,53],[832,285],[606,49],[595,274] )
#ProcessFrameData(bottominputfile2,[993,378],[312,378],[990,609],[317,609],[772,379],[538,385],[768,612],[538,618] )
#ProcessFrameData(topinputfile2,[999,137],[315,139],[995,369],[312,368],[768,134],[538,132],[778,371],[530,369] )
#ProcessFrameData(manual_20191108165756_bottom,[1071,303], [379,287], [1060,540],[372,520],[831,298],[594,295],[821,540],[598,535])
#ProcessFrameData(manual_20191108165756_top,[1073,54], [374,55], [1068,287],[374,274],[836,50],[599,50],[833,280],[593,50])
ProcessFrameData(manual_20191108170203_bottom,[993,378],[312,378],[990,609],[317,609],[772,379],[538,385],[768,612],[538,618] )
ProcessFrameData(manual_20191108170203_top,[999,137],[315,139],[995,369],[312,368],[768,134],[538,132],[778,371],[530,369])
#ProcessFrameData(manual_20191108143820_bottom,[1071,303], [379,287], [1060,540],[372,520],[831,298],[594,295],[821,540],[598,535])
#ProcessFrameData(manual_20191108143820_top,[1073,54], [374,55], [1068,287],[374,274],[836,50],[599,50],[833,280],[593,50])
#ProcessFrameData(manual_20191108152515_top,[1071,303], [379,287], [1060,540],[372,520],[831,298],[594,295],[821,540],[598,535])
#ProcessFrameData(manual_20191108152703_bottom, [993,378],[312,378],[990,609],[317,609],[772,379],[538,385],[768,612],[538,618])

print ("Building frames %s"%time.asctime())


frameDatanparray=[]
# build a frame data arrary after processing all files
frameDatanparray=np.array(frameData)       
df = pd.DataFrame(frameDatanparray[0:,0:],
                  columns=['frame num','ctorDist','ctolDist','voleRPositionInCountor','voleCPositionInCountor','voleLPositionInCountor',
                           'Cvelocity','Rvelocity','Lvelocity','Cacceleration','Racceleration','Lacceleration','CangofOrientation',
                           'RangofOrientation','LangofOrientation','CrawangofOrientation','RrawangofOrientation','LrawangofOrientation',
                           'ClongestDist','RlongestDist','LlongestDist','CavgDistanceFromCentroidToPoints','RavgDistanceFromCentroidToPoints',
                           'LavgDistanceFromCentroidToPoints','CcummulativeDist', 'RcummulativeDist','LcummulativeDist','Carea',
                           'Rarea','Larea','CX_V1','CY_V1','CMajorLength','CMinorLength' ,'RMajorLength','RMinorLength','LMajorLength',
                           'LMinorLength','Chamber' , 'CDistOUR', 'RDistOUR','LDistOUR','CDistOUL','RDistOUL',  'LDistOUL', 'CDistOLR',
                           'RDistOLR', 'LDistOLR','CDistOLL','RDistOLL', 'LDistOLL','CDistIUR','RDistIUR','LDistIUR','CDistIUL', 'RDistIUL',
                           'LDistIUL','CDistILR','RDistILR', 'LDistILR', 'CDistILL','RDistILL','LDistILL',
                           "CrotationalVel","RrotationalVel","LrotationalVel",
                           'Ceccentricity2','Reccentricity2','Leccentricity2','Ceccentricity1','Reccentricity1','Leccentricity1','Ceccentricitywdist',"Reccentricitywdist","Leccentricitywdist","Cclosetocentroid","Rclosetocentroid","Lclosetocentroid",
                           "ClongPtSlope", "RlongPtSlope","LlongPtSlope", "CTogetherFlag","RTogetherFlag","LTogetherFlag","CdistFromPreviousCont",
                          "RdistFromPreviousCont","LdistFromPreviousCont","CangleOfMovement","RangleOfMovement","LangleOfMovement","Careaoverlap","Rareaoverlap","Lareaoverlap"])

smoothing_window=11

#averages + smoothed for velocity, acceleration, distFromPreviousCont, major axis length and minor axis length

df['C_velocity_avg5'] = calcovingAvg(df["Cvelocity"], 5)
df['C_velocity_avg10'] = calcovingAvg(df["Cvelocity"], 10)
df['C_velocity_avg15'] = calcovingAvg(df["Cvelocity"], 15)
df['CSmoothened Velocity'] = scipy.signal.savgol_filter(df["Cvelocity"], smoothing_window, 1)

df['R_velocity_avg5'] = calcovingAvg(df["Rvelocity"], 5)
df['R_velocity_avg10'] = calcovingAvg(df["Rvelocity"], 10)
df['R_velocity_avg15'] = calcovingAvg(df["Rvelocity"], 15)
df['RSmoothened Velocity'] = scipy.signal.savgol_filter(df["Rvelocity"], smoothing_window, 1)


df['L_velocity_avg5'] = calcovingAvg(df["Lvelocity"], 5)
df['L_velocity_avg10'] = calcovingAvg(df["Lvelocity"], 10)
df['L_velocity_avg15'] = calcovingAvg(df["Lvelocity"], 15)
df['LSmoothened Velocity'] = scipy.signal.savgol_filter(df["Lvelocity"], smoothing_window, 1)

df['C_acceleration_avg5'] = calcovingAvg(df["Cacceleration"], 5)
df['C_acceleration_avg10'] = calcovingAvg(df["Cacceleration"], 10)
df['C_acceleration_avg15'] = calcovingAvg(df["Cacceleration"], 15)
df['CSmoothened Acceleration'] = scipy.signal.savgol_filter(df["Cacceleration"], smoothing_window, 1)

df['R_acceleration_avg5'] = calcovingAvg(df["Racceleration"], 5)
df['R_acceleration_avg10'] = calcovingAvg(df["Racceleration"], 10)
df['R_acceleration_avg15'] = calcovingAvg(df["Racceleration"], 15)
df['RSmoothened Acceleration'] = scipy.signal.savgol_filter(df["Racceleration"], smoothing_window, 1)


df['L_acceleration_avg5'] = calcovingAvg(df["Lacceleration"], 5)
df['L_acceleration_avg10'] = calcovingAvg(df["Lacceleration"], 10)
df['L_acceleration_avg15'] = calcovingAvg(df["Lacceleration"], 15)
df['LSmoothened Acceleration'] = scipy.signal.savgol_filter(df["Lacceleration"], smoothing_window, 1)


df['C_distFromPreviousCont_avg5'] = calcovingAvg(df["CdistFromPreviousCont"], 5)
df['C_distFromPreviousCont_avg10'] = calcovingAvg(df["CdistFromPreviousCont"], 10)
df['C_distFromPreviousCont_avg15'] = calcovingAvg(df["CdistFromPreviousCont"], 15)
df['CSmoothened distFromPreviousCont'] = scipy.signal.savgol_filter(df["CdistFromPreviousCont"], smoothing_window, 1)


df['R_distFromPreviousCont_avg5'] = calcovingAvg(df["RdistFromPreviousCont"], 5)
df['R_distFromPreviousCont_avg10'] = calcovingAvg(df["RdistFromPreviousCont"], 10)
df['R_distFromPreviousCont_avg15'] = calcovingAvg(df["RdistFromPreviousCont"], 15)
df['RSmoothened distFromPreviousCont'] = scipy.signal.savgol_filter(df["RdistFromPreviousCont"], smoothing_window, 1)


df['L_distFromPreviousCont_avg5'] = calcovingAvg(df["LdistFromPreviousCont"], 5)
df['L_distFromPreviousCont_avg10'] = calcovingAvg(df["LdistFromPreviousCont"], 10)
df['L_distFromPreviousCont_avg15'] = calcovingAvg(df["LdistFromPreviousCont"], 15)
df['LSmoothened distFromPreviousCont'] = scipy.signal.savgol_filter(df["RdistFromPreviousCont"], smoothing_window, 1)

df['CMajorLength_avg5'] = calcovingAvg(df["CMajorLength"], 5)
df['CMajorLength_avg10'] = calcovingAvg(df["CMajorLength"], 10)
df['CMajorLength_avg15'] = calcovingAvg(df["CMajorLength"], 15)

df['RMajorLength_avg5'] = calcovingAvg(df["RMajorLength"], 5)
df['RMajorLength_avg10'] = calcovingAvg(df["RMajorLength"], 10)
df['RMajorLength_avg15'] = calcovingAvg(df["RMajorLength"], 15)

df['LMajorLength_avg5'] = calcovingAvg(df["LMajorLength"], 5)
df['LMajorLength_avg10'] = calcovingAvg(df["LMajorLength"], 10)
df['LMajorLength_avg15'] = calcovingAvg(df["LMajorLength"], 15)

df['LMinorLength_avg5'] = calcovingAvg(df["CMinorLength"], 5)
df['LMinorLength_avg10'] = calcovingAvg(df["CMinorLength"], 10)
df['LMinorLength_avg15'] = calcovingAvg(df["CMinorLength"], 15)

df['RMinorLength_avg5'] = calcovingAvg(df["RMinorLength"], 5)
df['RMinorLength_avg10'] = calcovingAvg(df["RMinorLength"], 10)
df['RMinorLength_avg15'] = calcovingAvg(df["RMinorLength"], 15)

df['LMinorLength_avg5'] = calcovingAvg(df["LMinorLength"], 5)
df['LMinorLength_avg10'] = calcovingAvg(df["LMinorLength"], 10)
df['LMinorLength_avg15'] = calcovingAvg(df["LMinorLength"], 15)

# Write positions to file
df.to_csv(output_filepath, sep=',')

print("completed the frames all done %s" % len(frameDatanparray))


# In[24]:


#LOOKING AT BEHAVIORS

import matplotlib.pyplot as plt
import numpy as np

def getMeanandStdev(myarray, behavior):
    # convert to NP array
    NPArray=np.array(myarray)  
    return ([behavior, np.mean(NPArray), np.std(NPArray)]) 


manualdf= pd.read_csv("//Users//aakritilakshmanan//Downloads//110819_Shift1_VC3_Lane2.csv", header=0)
manual20191108165756_top = pd.read_csv("//Users//aakritilakshmanan//Downloads//20191108165756_top.csv", header=0)
manual20191108152515_top = pd.read_csv("//Users//aakritilakshmanan//Downloads//20191108152515_top.csv", header=0)
manual20191108143820_top = pd.read_csv("//Users//aakritilakshmanan//Downloads//20191108143820_top.csv", header=0)


plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)

# calculating the duration of specific behaviors
def calcDuration(file):
    columns=['Left', 'Center', 'Right', 'Interact Right', 'Interact Left', 'Huddle Left', "Attack Right", "Huddle Right"]
    global leftArray
    global CenterArray
    global RightArray
    global InteractRightArray
    global InteractLeftArray
    global HuddleLeftArray
    global AttackRightArray
    global HuddleRightArray

    length = file.shape[0]

    for i in range(0, length-1):
        if file.iloc[i,5] == file.iloc[i+1,5]:
            if file.iloc[i,5] == "Left":
                leftArray.append(file.iloc[i+1,0]-file.iloc[i,0])
            elif file.iloc[i,5] == "Attack Right":
                AttackRightArray.append(file.iloc[i+1,0]-file.iloc[i,0])
            elif file.iloc[i,5] == "Center":
                CenterArray.append(file.iloc[i+1,0]-file.iloc[i,0])
            elif file.iloc[i,5] == "Huddle Left":
                HuddleLeftArray.append(file.iloc[i+1,0]-file.iloc[i,0])
            elif file.iloc[i,5] == "Huddle Right":
                HuddleRightArray.append(file.iloc[i+1,0]-file.iloc[i,0])
            elif file.iloc[i,5] == "Interact Left":
                InteractLeftArray.append(file.iloc[i+1,0]-file.iloc[i,0])
            elif file.iloc[i,5] == "Interact Right":
                InteractRightArray.append(file.iloc[i+1,0]-file.iloc[i,0])
            elif file.iloc[i,5] == "Right":
                RightArray.append(file.iloc[i+1,0]-file.iloc[i,0])

leftArray = []
CenterArray=[]
RightArray=[]
InteractRightArray=[]
InteractLeftArray=[]
HuddleLeftArray=[]
AttackRightArray=[]
HuddleRightArray=[]

calcDuration(manualdf)
calcDuration(manual20191108165756_top)
calcDuration(manual20191108152515_top)
calcDuration(manual20191108143820_top)
columns=['Left', 'Center', 'Right', 'Interact Right', 'Interact Left', 'Huddle Left', "Attack Right", "Huddle Right"]

finalArray=[]                

finalArray.append(getMeanandStdev(leftArray,columns[0]))
finalArray.append(getMeanandStdev(CenterArray,columns[1]))
finalArray.append(getMeanandStdev(RightArray,columns[2]))
finalArray.append(getMeanandStdev(InteractRightArray,columns[3]))
finalArray.append(getMeanandStdev(InteractLeftArray,columns[4]))
finalArray.append(getMeanandStdev(HuddleLeftArray,columns[5]))
finalArray.append(getMeanandStdev(AttackRightArray,columns[6]))
#finalArray.append(getMeanandStdev(HuddleRightArray,columns[7]))


columns=['Left', 'Center', 'Right', 'Interact Right', 'Interact Left', 'Huddle Left', "Attack Right"]
x_pos = np.arange(len(columns))
error = (finalArray[0][2], finalArray[1][2], finalArray[2][2],finalArray[3][2],finalArray[4][2],finalArray[5][2],finalArray[6][2])
means = (finalArray[0][1], finalArray[1][1], finalArray[2][1],finalArray[3][1],finalArray[4][1],finalArray[5][1],0)

#plot avg length
plt.figure()
plt.bar(columns, means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10, color="#6D99B4")
plt.xticks(x_pos, columns, fontsize=16)
plt.xticks(rotation=90)
plt.yticks(fontsize=13)
plt.ylabel('Length (seconds)', fontsize=16)
plt.title('Behavior Average Length', fontsize=16) 
plt.savefig('lengthofBehavior.png',format='png', dpi=300, bbox_inches="tight")


def calcPercentage (array):
    percentage = sum(array)/((84728+84571+9786+82349)/30)
    return 100*percentage

#plot percentage
objects = ['Left', 'Center','Right','Interact Right','Interact Left','Huddle Left', "Attack Right"]
percentages = (calcPercentage(leftArray),calcPercentage(CenterArray), calcPercentage(RightArray),calcPercentage(InteractRightArray),calcPercentage(InteractLeftArray),calcPercentage(HuddleLeftArray), calcPercentage(AttackRightArray))          
            
plt.figure()
plt.bar(columns, percentages, align='center', alpha=0.5, color="#6D99B4")
plt.xticks(x_pos, objects, fontsize=16)
plt.xticks(rotation=90)
plt.yticks(fontsize=13)
plt.ylabel('Percentage',fontsize=16)
plt.title('Percentage of Time Spent in Behaviors',fontsize=16) 
plt.savefig('percentageofBehavior.png',format='png', dpi=300, bbox_inches="tight")


# In[17]:


#file processing only no charts and images 
import pandas as pd

manual_20191108161215_bottom = pd.read_csv("//Users//aakritilakshmanan//Downloads//110819_Shift1_VC3_Lane1.csv", header=0)
manual_20191108161215_top = pd.read_csv("//Users//aakritilakshmanan//Downloads//110819_Shift1_VC3_Lane2.csv", header=0)
manual_20191108161442_bottom = pd.read_csv("//Users//aakritilakshmanan//Downloads//110819_Shift1_VC1_Lane3.csv", header=0)
manual_20191108161442_top = pd.read_csv("//Users//aakritilakshmanan//Downloads//110819_Shift1_VC1_Lane4.csv", header=0)

manual20191108165756_bottom = pd.read_csv("//Users//aakritilakshmanan//Downloads//20191108165756_bottom.csv", header=0)
manual20191108165756_top = pd.read_csv("//Users//aakritilakshmanan//Downloads//20191108165756_top.csv", header=0)
manual20191108170203_bottom = pd.read_csv("//Users//aakritilakshmanan//Downloads//20191108170203_bottom.csv", header=0)
manual20191108170203_top = pd.read_csv("//Users//aakritilakshmanan//Downloads//20191108170203_top.csv", header=0)
manual20191108143820_bottom = pd.read_csv("//Users//aakritilakshmanan//Downloads//20191108143820_bottom.csv", header=0)
manual20191108143820_top = pd.read_csv("//Users//aakritilakshmanan//Downloads//20191108143820_top.csv", header=0)
manual20191108152515_top = pd.read_csv("//Users//aakritilakshmanan//Downloads//20191108152515_top.csv", header=0)
manual20191108152703_bottom = pd.read_csv("//Users//aakritilakshmanan//Downloads//20191108152703_bottom.csv", header=0)


bottominputfile = "//Users//aakritilakshmanan//Desktop//vole_contour_data//manual_20191108161215.bottom.resolved.contours"
topinputfile = "//Users//aakritilakshmanan//Desktop//vole_contour_data//manual_20191108161215.top.resolved.contours.csv"
bottominputfile2 = "//Users//aakritilakshmanan//Desktop//vole_contour_data//manual_20191108161442.bottom.resolved.contours"
topinputfile2 = "//Users//aakritilakshmanan//Desktop//vole_contour_data//manual_20191108161442.top.resolved.contours"
manual20191108165756_bottomCntr= "//Users//aakritilakshmanan//Downloads//manual_20191108165756.bottom.resolved.contours"
manual20191108165756_topCntr="//Users//aakritilakshmanan//Downloads//manual_20191108165756.top.resolved.contours"
manual20191108170203_bottomCntr= "//Users//aakritilakshmanan//Downloads//manual_20191108170203.bottom.resolved.contours"
manual20191108170203_topCntr="//Users//aakritilakshmanan//Downloads//manual_20191108170203.top.resolved.contours"
manual20191108143820_bottomCntr= "//Users//aakritilakshmanan//Downloads//manual_20191108143820.bottom.resolved.contours"
manual20191108143820_topCntr= "//Users//aakritilakshmanan//Downloads//manual_20191108143820.top.resolved.contours"
manual20191108152515_topCntr= "//Users//aakritilakshmanan//Downloads//manual_20191108152515.top.resolved.contours"
manual20191108152703_bottomCntr= "//Users//aakritilakshmanan//Downloads//manual_20191108152703.bottom.resolved.contours"



# associating behaviors to frame numbers

def createHotEncoding (manualfile, ctrFile):
    
    recordCount = len(manualfile)
    #print("farme count %s"% recordCount)
    
    # adding frmae number based on time multiplied by 30 frames per second for each rows in manual annotation
    manualfile['frame num']=round(manualfile.iloc[:,0]*30,0)
    
    #find the max frame count do this after calculation
    FrameCount = 0
    with open(ctrFile, "r") as f:
        FrameCount = ( sum(1 for line in f) )-1
    
    #print("farme count %s"% FrameCount)
    
    columns=['Left', 'Center', 'Right', 'Interact Right', 'Interact Left', 'Huddle Left', "Attack Right", "Huddle Right"]
    bdf=pd.DataFrame(index=range(0,FrameCount),columns=columns)
    bdf = bdf.fillna(0)
    
    
    if manualfile.iloc[1,5]== 'Huddle Left':
        bdf.at[0:int(manualfile.iloc[1,9]),'Huddle Left']= 1
    if manualfile.iloc[1,5]== "Huddle Right":
        bdf.at[0:int(manualfile.iloc[1,9]),"Huddle Right"]= 1
    if manualfile.iloc[recordCount-1,5]== "Huddle Right":
        bdf.at[int(manualfile.iloc[int(recordCount-1),9]):int(FrameCount),"Huddle Right"]= 1
    if manualfile.iloc[recordCount-1,5]== 'Huddle Left':
        bdf.at[int(manualfile.iloc[int(recordCount-1),9]):int(FrameCount),'Huddle Left']= 1

    for i in range(0,recordCount-1):
        if manualfile.iloc[i,5] == manualfile.iloc[i+1,5]:
            for k in range(len(columns)):
                if manualfile.iloc[i,5] == columns[k]:
                    bdf.at[int(manualfile.iloc[i,9]):int(manualfile.iloc[i+1,9]),columns[k]]= 1
    return bdf

#put em together!
final= pd.concat([createHotEncoding(manual_20191108161215_bottom, bottominputfile),
                  createHotEncoding(manual_20191108161215_top, topinputfile),
                  createHotEncoding(manual_20191108161442_bottom, bottominputfile2),
                  createHotEncoding(manual_20191108161442_top, topinputfile2),
                  createHotEncoding(manual20191108165756_bottom, manual20191108165756_bottomCntr),
                  createHotEncoding(manual20191108165756_top, manual20191108165756_topCntr),
                  createHotEncoding(manual20191108170203_bottom, manual20191108170203_bottomCntr),
                  createHotEncoding(manual20191108170203_top, manual20191108170203_topCntr),
                  createHotEncoding(manual20191108143820_bottom, manual20191108143820_bottomCntr),
                  createHotEncoding(manual20191108143820_top, manual20191108143820_topCntr),
                  createHotEncoding(manual20191108152515_top, manual20191108152515_topCntr ),
                  createHotEncoding(manual20191108152703_bottom, manual20191108152703_bottomCntr)
                 ])


final.reset_index(level=0, inplace=True)   
final = final.rename(columns={'index': 'frame num'})
final["huddle total"] = 0

for i in range(len(final)):
    if final.iloc[i,6] ==1 or final.iloc[i,8] ==1:
        final.at[i,'huddle total'] = 1
    else:
        final.at[i,'huddle total'] = 0

final.to_csv("//Users//aakritilakshmanan//Desktop//vole_contour_data//finalallbad.aakriti.csv" , sep=',')  
print("Finished")


# In[9]:


#getting a figure for each parameter

x = range(0,2000)
left = bdf.iloc[0:2000,0]
center = bdf.iloc[0:2000,1]
right = bdf.iloc[0:2000,2]
intright = bdf.iloc[0:2000,3]
intleft = bdf.iloc[0:2000,4]
huddleleft = bdf.iloc[0:2000,5]
attackright = bdf.iloc[0:2000,6]
huddleright = bdf.iloc[0:2000,7]


f, (ax1, ax2, ax3, ax4, ax5,ax6,ax7,ax8,ax9) = plt.subplots(9, sharex=True, sharey=False)
ax1.bar(x, left)
ax1.set_ylabel(r'Left', size =10)
ax1.set_title('Features')
ax2.bar(x, center, color = 'g')
ax2.set_ylabel(r'Center', size =10)

ax3.bar(x, right, color='r')
ax3.set_ylabel(r'Right', size =10)

ax4.bar(x, intright, color='r')
ax4.set_ylabel(r'Interact Right', size =10)

ax5.bar(x, intleft, color='r')
ax5.set_ylabel(r'Int Left', size =10)

ax6.bar(x, huddleleft, color='r')
ax6.set_ylabel(r'Huddle Left', size =10)

ax7.plot(x,  frameDatanparray[0:2000,6])
ax7.set_ylabel(r'Velocity', size =10)

ax8.plot(x,  frameDatanparray[0:2000,27])
ax8.set_ylabel(r'Area', size =10)

f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
ax1.set_axis_off
plt.setp([a.get_yticklabels() for a in f.axes[:4]], visible=False)


# In[139]:


#plotting the endpoints onto the video 

import ast
import copy
import time
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import pickle

min_area = 1000
max_area = 20000
scaling = 1.0
# multiple tracking
mot = True

video = 'manual_20191108161442_first_min'
upper_left_frame = (250, 100)
lower_right_frame = (1050, 373)

poly = True
roi_corners = np.array([[(328,12), (1115,15), (1118,294), (310,278)]], dtype=np.int32)

start_time = 1000000
end_time = 30
late_start = False
early_end = False
threshold = 80
fps = 30

input_vidpath = '/Users/aakritilakshmanan/Documents/vidddd1.m4v'
output_vidpath = '/Users/aakritilakshmanan/Downloads/manual_20191108161215_plotted_botton.mp4'
codec = 'DIVX'

cap = cv2.VideoCapture(input_vidpath)
if not cap.isOpened():
    print("Cannot open video file")
    exit()

fourcc = cv2.VideoWriter_fourcc(*codec)
output_framesize = (int(cap.read()[1].shape[1]*scaling),int(cap.read()[1].shape[0]*scaling))
out = cv2.VideoWriter(filename = output_vidpath, fourcc = fourcc, fps = 30.0, frameSize = output_framesize, isColor = True)

print("starting")
i=0
while True: 
    
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    final = frame.copy()
    #cv2.circle(final,  tuple([int(x) for x in frameDatanparray[i+1,38][0] ]) , 5, (0, 0, 255), -1, cv2.LINE_AA)
    #cv2.circle(final,  tuple([int(x) for x in frameDatanparray[i+1,38][1] ]) , 5, (0, 255, 0), -1, cv2.LINE_AA)
    #cv2.circle(final,  tuple([int(x) for x in frameDatanparray[i+1,39][0] ]) , 5, (255, 0, 0), -1, cv2.LINE_AA)
    #cv2.circle(final,  tuple([int(x) for x in frameDatanparray[i+1,39][1] ]) , 5, (255, 255, 0), -1, cv2.LINE_AA)
    #cv2.circle(final,  tuple([int(frameDatanparray[i+1,30]),int(frameDatanparray[i+1,31])]) , 5, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(final,  tuple([int(x) for x in frameDatanparray[i+1,71]]) , 5, (255,0, 0), -1, cv2.LINE_AA)
    cv2.circle(final,  tuple([int(x) for x in frameDatanparray[i+1,70]]) , 5, (255, 152, 255), -1, cv2.LINE_AA)
    #cv2.putText(final, tuple([, (5,700), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,0,0), 2)
    #cv2.putText(final,tuple([int(x) for x in frameDatanparray[i,38][0] ]),(10,500), font, 1,(255,255,255),2)
    
    if cv2.waitKey(1) == ord('q'):
        break
    out.write(final)
    
    i=i+1
    if(i==1000):
        break

# When everything done, release the capture
print("finished")
cap.release()
out.release()
cv2.destroyAllWindows()


# In[1]:


#FEATURES STILL IN PROGRESS

#CURVATURE
   #rotate and align the blob vertically
    """
        theta = np.arctan((self.x_v1)/(self.y_v1)) 
        rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        transformed_centroid=rotation_mat*[[self.cx],[self.cy]]
        transformed_mat = rotation_mat*actualMatrix
        x_transformed, y_transformed = transformed_mat
        
        
        xlabel = np.array( transformed_mat[0] ) 
        ylabe1 = np.array ( transformed_mat[1] )
        centroidarray= np.array(transformed_centroid)
        
        xyarray = []
        for i in range(len(xlabel[0])):
            xyarray.append([xlabel[0][i]-centroidarray[0][0],ylabe1[0][i]-centroidarray[1][0]])
        
        #split coordinates into two curves
        for i in range(len(xyarray)):
            if xyarray[i][1] >= 0:
                topcurve.append([xyarray[i][0],xyarray[i][1]])
            else:
                bottomcurve.append([xyarray[i][0],xyarray[i][1]])
        
        
        print(calcAnglewg3(self.majorPt[0],self.minorPt[0],[self.cx,self.cy]))
        print(calcAnglewg3(self.majorPt[1],self.minorPt[0],[self.cx,self.cy]))
        
        
        #if np.std(self.calcCurvature(np.array(topcurve)))>np.std(self.calcCurvature(np.array(bottomcurve))):
            #print("top")
        #else:
            #print("bottom")
        """
            
#calculating tortuisity
#areas where the angle deviates the most from 180 are the areas where there are more turns and twists
"""
tortuisityL=[]
tortuisityR = []
tortuisityC =[]
for frameCount in range(len(frames)-3):
    tortuisityfullL = []
    tortuisityfullR = []
    tortuisityfullC = []
    rAngle=0.0
    rAngle= calcAnglew3 (frames[frameCount].contours[frames[frameCount].voleRPositionInCountor], 
                frames[frameCount+1].contours[frames[frameCount+1].voleRPositionInCountor], 
                frames[frameCount+2].contours[frames[frameCount+2].voleRPositionInCountor])
    tortuisityfullR.append(rAngle)
    tortuisityR.append(sum(tortuisityfullR)/(2*math.pi)) 
    
tortuisityRnp=np.array(tortuisityR)
np.amax(tortuisityRnp)
"""


# In[ ]:




