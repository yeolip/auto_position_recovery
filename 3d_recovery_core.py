import numpy as np
import os
import math
import cv2
import sys
import copy
import csv
import pandas as pd
import threading
from itertools import combinations
import time
import datetime as dt

import tkinter as tk
import tkinter.ttk
# from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

import openpyxl
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# 12 GT점
# 548.0606, -0.9230, 202.5364,   C
# 548.1446, -125.9743, 202.7360, N
# 548.1358, 123.9789, 202.4794,
# 658.2693, -0.8927, 203.8425,   C
# 658.2218, -157.4343, 204.1222, N
# 658.4669, 155.4454, 203.8754,
# 768.5166, -0.9942, 205.2305,   C
# 768.3877, -187.8320, 205.2665, N
# 768.5165, 186.0766, 205.2247,  P
# 998.9362, -0.9115, 207.5076,   C
# 999.4603, -223.7147, 208.0781, N(L)
# 998.7845, 222.0334, 207.5687   P(R)

#Automatic virtual position recovery using relative coordinates

# 최대 줄 수 설정
pd.set_option('display.max_rows', 2500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 50)
# 표시할 가로의 길이
pd.set_option('display.width', 160)
# 출력값 소숫점4자리로 설정
pd.options.display.float_format = '{:.4f}'.format

C_TAB1 = 0
C_TAB2 = 1
C_TAB3 = 2
C_TAB4 = 3
C_TAB5 = 4

C_MAX_GAP = 2       # 중복좌표 체크 범위 (단위 mm)
C_MIN_VALUE_RIGID_CALC = 3
C_PRINT_ALGO_ENABLE = 0
C_PRINT_CTRL_ENABLE = 0

degreeToRadian = math.pi/180
radianToDegree = 180/math.pi

def print_current_time(text=''):
    tnow = dt.datetime.now()
    print('%s-%2s-%2s %2s:%2s:%2s \t%s' % (tnow.year, tnow.month, tnow.day, tnow.hour, tnow.minute, tnow.second, text))


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    # sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    sy = math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])

    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([ x, y, z])

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

#숫자 자릿수 리턴
def digit_length(n):
    ans = 0
    while n:
        n //= 10
        ans += 1
    return ans

def funcname():
    return sys._getframe(1).f_code.co_name + "()"

def callername():
    return sys._getframe(2).f_code.co_name + "()"

class RecoveryAlgo():
    def __del__(self):
        print("*************delete RecoveryAlgo class***********\n")
    def __init__(self):
        self.debugflag = C_PRINT_ALGO_ENABLE
        print("*************initialize RecoveryAlgo class***********\n")

    def m_ProjectUserCoor(self, input_point, ref):
        """
        A function to project user coordinates based on input points and a reference.
        
        Parameters:
        - input_point: array-like, shape (n, 3)
            The input points to be projected.
        - ref: array-like, shape (5, 3)
            The reference points for the projection.
        
        Returns:
        - output_point: array-like, shape (n, 3)
            The projected output points based on the input points and reference.
        - t_mean: array-like, shape (3,)
            The mean of the reference points.
        - t_axis: array-like, shape (3, 3)
            The transformation axis matrix.
        """
        print("//////////", funcname(), "//////////")

        # t_mean = np.mean(ref, axis= 0);
        t_mean = np.mean((ref[0,:],ref[1,:],ref[3,:],ref[4,:]), axis=0)

        # AE = t_mean - ref[0,:]
        # BE = ref[1,:] - t_mean;
        #
        # XAxis = np.cross(AE, BE);
        # YAxis = np.cross(AE, XAxis);
        # ZAxis = np.cross(XAxis, YAxis);

        P0 = ref[0,:] - t_mean
        P1 = ref[1,:] - t_mean

        YAxis = np.cross(P0, P1);
        ZAxis = np.cross(P0, YAxis);
        XAxis = np.cross(YAxis, ZAxis);

        XAxis = XAxis / np.linalg.norm(XAxis);
        YAxis = YAxis / np.linalg.norm(YAxis);
        ZAxis = ZAxis / np.linalg.norm(ZAxis);

        t_axis = np.eye(3)
        t_axis[:, 0] = XAxis
        t_axis[:, 1] = YAxis
        t_axis[:, 2] = ZAxis
        print('mean', t_mean, '\nt_axis', t_axis, sep='\n')

        print(rotationMatrixToEulerAngles(t_axis) * radianToDegree)
        # print(eulerAnglesToRotationMatrix(rotationMatrixToEulerAngles(t_axis)) )
        print(cv2.Rodrigues(t_axis)[0] * radianToDegree)
        # print(cv2.Rodrigues(cv2.Rodrigues(t_axis)[0]) )
        # output_point = (input_point - np.tile(t_mean, (input_point.shape[0],1))) * t_axis
        output_point = (input_point - np.tile(t_mean, (input_point.shape[0],1))) * t_axis

        if (self.debugflag):
            print('ret',*output_point, sep='\n')
        return output_point, t_mean, t_axis


    def m_ProjectAbsCoor(self, P_DPA_Pts, P_DPA_Ref1):
        """
        A function to calculate the absolute coordinates of points based on given reference points.
        
        Parameters:
        - P_DPA_Pts: numpy matrix representing points in DPA coordinates
        - P_DPA_Ref1: numpy matrix representing the reference points in DPA coordinates
        
        Returns:
        - P_Ref1_Pts: numpy matrix containing the absolute coordinates of the points
        """
        print("//////////", funcname(), "//////////")
        # P_DPA_Pts = np.asmatrix([[391.139000000000, - 1399.38500000000,    706.558000000000],
        # [318.651000000000, - 1399.80100000000,    481.105000000000],
        # [183.801000000000, - 1399.43800000000,    525.289000000000],
        # [203.573000000000, - 1459.04200000000,    645.357000000000],
        # [259.687000000000, - 1399.01000000000,    749.782000000000]])
        #
        # P_DPA_Ref1 = np.asmatrix([[949.852000000000, - 51.6980000000000,    731.117000000000],
        # [727.499000000000, - 53.3340000000000,    42.3240000000000],
        # [451.291000000000, - 53.0250000000000,    133.457000000000],
        # [646.216000000000, - 51.4270000000000,    831.055000000000]])

        Centroid, Axis = self.m_MakeAbsAxis(P_DPA_Ref1);
        # print(P_DPA_Pts, np.tile(Centroid, (P_DPA_Pts.shape[0],1)),'\n',Axis )
        P_Ref1_Pts = (P_DPA_Pts - np.tile(Centroid, (P_DPA_Pts.shape[0],1))) * Axis

        if (self.debugflag):
            print('ret',*P_Ref1_Pts, sep='\n')
        # P_Ref1_Pts =
        # 1347.73296211896    404.477604521447 - 5.85905452814663
        # 1347.61341605657    311.995327466233    212.156432418129
        # 1347.38468857793    442.937470819739    266.846311762761
        # 1407.27742564632    506.339726820845    163.150085194833
        # 1347.48922525842    532.220488679192    47.3366148858555
        # np.tile(T, (1, P_Ref.shape[0]))
        return P_Ref1_Pts


    def m_ProjectDispCoor(self, P_Ref1_Pts, P_Ref1_Disp):
        """
        A function that calculates the displacement of reference points based on given reference points and displacements. Returns the displaced points, centroid, and axis.

        Parameters:
        - P_Ref1_Pts: A matrix of reference points.
        - P_Ref1_Disp: A matrix of displacements for the reference points.

        Returns:
        - P_Disp: Displaced points based on reference points, centroid, and axis.
        - Centroid: The centroid of the displaced points.
        - Axis: The axis of the displaced points.
        """
        print("//////////", funcname(), "//////////")
        # P_Ref1_Pts = np.asmatrix(
        # [[1261.29623616261,    239.985317917590,    51.5313160178364],
        # [1261.53702655582,    154.082122583680,    257.056595465577],
        # [1261.50379097605,    325.570613718301, - 154.102117931721],
        # [1031.00674552654,    239.708176629555,    51.3230778612638],
        # [1031.10349410293,    167.687936469370,    223.972390869635],
        # [1030.97732105302,    311.481917956706, - 121.116009669177],
        # [930.758616411388,    239.605145729361,    51.2910088561620],
        # [930.840679549455,    179.348376193649,    195.831057431068],
        # [930.703960598259,    299.751880130932, - 93.1282020090007],
        # [820.687625912447,    239.565695359699,    51.3331018939299],
        # [820.678623657056,    191.501615649785,    166.651377890795],
        # [820.585820231081,    287.700179385780, - 64.0824243331095]]
        # )
        #
        # P_Ref1_Disp = np.asmatrix(
        # [[261.0658,    430.6803, - 15.0083],
        # [261.1287,    326.2123,    234.2476],
        # [260.3185,    407.3123,    268.2431],
        # [260.2502,    511.8065,    18.9925],
        # [260.6193,    418.9697,    126.6069]]
        # )
        Centroid, Axis = self.m_MakeDispAxis(P_Ref1_Disp);

        # Centroid = np.asmatrix([260.6765,  418.9962,  126.6164])
        # Axis = np.asmatrix(
        # [[0.999957259960338, 2.426512344112703e-04, 0.009242267266263],
        # [0.008617504541717, -0.386586756527017, -0.922224100117558],
        # [0.003349159310558, 0.922253013439533, -0.386545326783471]]
        # )
        P_Disp = (P_Ref1_Pts - np.tile(Centroid, (P_Ref1_Pts.shape[0],1))) * Axis

        if (self.debugflag):
            print('ret',P_Disp)
        return P_Disp, Centroid, Axis


    def rigid_transform_3D(self, A, B):
        """
        A function to calculate the rigid transformation between two sets of 3D points.
        
        Parameters:
        - A: numpy array of shape (N, 3) representing the first set of 3D points
        - B: numpy array of shape (N, 3) representing the second set of 3D points
        
        Returns:
        - R: the rotation matrix for the transformation
        - t: the translation vector for the transformation
        """
        print("//////////",funcname(),"//////////")
        assert len(A) == len(B)

        N = A.shape[0];  # total points

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # centre the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))

        # dot is matrix multiplication for array
        H = np.transpose(AA) * BB

        U, S, Vt = np.linalg.svd(H)

        R = Vt.T * U.T

        # special reflection case
        if np.linalg.det(R) < 0:
            print("Reflection detected")
            Vt[2, :] *= -1
            R = Vt.T * U.T

        t = -R * centroid_A.T + centroid_B.T
        # print(t)

        return R, t

    def m_findTransformedPoints(self, P_Ref, P_Input, P_Target):
        """
        A function to find transformed points based on input and reference points.

        Parameters:
        - P_Ref: A numpy matrix representing reference points.
        - P_Input: A numpy matrix representing input points.
        - P_Target: A numpy matrix representing target points.

        Returns:
        - P_Transformed: A numpy matrix of transformed points.
        - R: A numpy matrix representing rotation.
        - T: A numpy matrix representing translation.
        """
        print("//////////",funcname(),"//////////")
        # P_Ref = np.asmatrix(
        # [[1206.58800000000, - 52.3840000000000,    845.141000000000],
        # [869.521000000000, - 44.0830000000000, - 206.674000000000],
        # [-25.2610000000000, - 59.8710000000000,    56.7040000000000],
        # [308.079000000000, - 68.3820000000000,    1105.31600000000]]
        # )
        #
        # P_Input = np.asmatrix(
        # [[575.621000000000, - 53.3910000000000,    347.671000000000],
        # [1296.75700000000, - 46.9340000000000, - 489.023000000000],
        # [604.683000000000, - 44.9410000000000, - 1114.64600000000],
        # [-115.650000000000, - 51.2140000000000, - 282.811000000000]]
        # )
        #
        # P_Target = np.asmatrix([[943.3570, - 54.7900,    731.0520],
        # [727.7780, - 48.9640,    40.1490],
        # [450.7460, - 53.8990,    128.5110],
        # [638.8010, - 60.2540,    827.9780]])

        # print('P_Input.shape[0],P_Ref.shape[0]', P_Input.shape[0],P_Ref.shape[0] )
        n_loop = P_Input.shape[0] / P_Ref.shape[0]
        # print(n_loop, P_Input.shape[0],P_Ref.shape[0] )
        # n_Ref = P_Ref.shape[0]

        P_Transformed = [];

        for i in range(int(n_loop)):

            # [R, T] = rigid_transform_3D(P_Ref', P_Input(n_Ref*(i-1)+1:n_Ref*i,:)');

            # R = np.asmatrix([[0.522094256732165, 0.015812569675465, -0.852741197390830],
            # [-0.017819784736575, 0.999812105327660, 0.007629502750195],
            # [0.852701613906782, 0.011212345005914, 0.522277934590614]])
            #
            # T = np.asmatrix([6.671829005434147e+02, 14.124131785083954, -1.122026213358952e+03])

            R, T = self.rigid_transform_3D(P_Ref, P_Input)

            print('R',*R, 'T',*T , sep='\n')
            # print(P_Target, P_Target.shape)
            # P_Transformed = [P_Transformed;(R * P_Target'+T)'];
            P_Transformed = (R * P_Target.T + T).T
            # P_Transformed = (R * P_Target.T) + np.tile(T, (1, P_Ref.shape[0]))
            # P_Transformed = P_Transformed.T
            # P_Transformed = (R * P_Target[0].T + T.T )
            # P_Transformed[1] = (R * P_Target[1].T + T.T )
            # P_Transformed[2] = (R * P_Target[2].T + T.T )

            if(self.debugflag):
                print('ret', *P_Transformed, sep='\n')
            # print('ret', P_Transformed)
            # P_Transformed =
            # 535.439643764020, - 51.8884288910241,  63.5738272867725
            # 1012.14066152380, - 47.4931985302768, - 481.028802658092
            # 792.075892677580, - 46.8164625429134, - 671.160246228224
            # 293.693512131688, - 51.1847826907368, - 145.560518603204
        return P_Transformed, R, T

    # OK
    def m_MakeAbsAxis(self, P_DPA_Ref1):
        """
        A function to calculate the centroid and axis of a given array.

        Parameters:
        - P_DPA_Ref1: a numpy array representing points in space

        Returns:
        - centroid: a numpy array representing the centroid of the input points
        - axis: a numpy array representing the axis of the input points
        """
        print("//////////",funcname(),"//////////")
        # P_DPA_Ref1 = np.array([[949.852000000000, - 51.6980000000000,    731.117000000000],
        # [727.499000000000, - 53.3340000000000,    42.3240000000000],
        # [451.291000000000, - 53.0250000000000,    133.457000000000],
        # [646.216000000000, - 51.4270000000000,    831.055000000000]])

        centroid = np.mean(P_DPA_Ref1, axis= 0);
        AE = centroid - P_DPA_Ref1[0,:]
        # -256.137500000000	-0.673000000000002	-296.628750000000
        BE = P_DPA_Ref1[1,:] - centroid;

        XAxis = np.cross(AE, BE);
        YAxis = np.cross(AE, XAxis);
        ZAxis = np.cross(XAxis, YAxis);

        XAxis = XAxis / np.linalg.norm(XAxis);
        YAxis = YAxis / np.linalg.norm(YAxis);
        ZAxis = ZAxis / np.linalg.norm(ZAxis);

        axis = np.eye(3)
        axis[:, 0] = XAxis
        axis[:, 1] = YAxis
        axis[:, 2] = ZAxis

        print('mean', centroid, '\naxis', axis, sep='\n')
        # centroid 693.714500 -52.371000 434.488250
        # XAxis -0.000197 -0.999997 0.002439
        # YAxis -0.756877 0.001743 0.653555
        # ZAxis -0.653558 -0.001717 -0.756875

        return centroid, axis

    def m_MakeDispAxis(self, P_Ref1_Disp):
        """
        Generate the centroid and axis of a given set of points in 3D space.

        Parameters:
        - P_Ref1_Disp (numpy.ndarray): The set of points in 3D space.

        Returns:
        - centroid (numpy.ndarray): The centroid of the set of points.
        - axis (numpy.ndarray): The axis of the set of points.
        """
        print("//////////",funcname(),"//////////")
        centroid = np.mean(P_Ref1_Disp, axis= 0)

        L_C = (P_Ref1_Disp[0,:] + P_Ref1_Disp[3,:]) / 2
        R_C = (P_Ref1_Disp[1,:] + P_Ref1_Disp[2,:]) / 2
        C_T = (P_Ref1_Disp[0,:] + P_Ref1_Disp[1,:]) / 2
        C_B = (P_Ref1_Disp[2,:] + P_Ref1_Disp[3,:]) / 2

        YAxis = R_C - L_C
        ZAxis = C_T - C_B
        XAxis = np.cross(YAxis, ZAxis)

        XAxis = XAxis / np.linalg.norm(XAxis)
        YAxis = YAxis / np.linalg.norm(YAxis)
        ZAxis = ZAxis / np.linalg.norm(ZAxis)

        axis = np.eye(3)
        axis[:, 0] = XAxis
        axis[:, 1] = YAxis
        axis[:, 2] = ZAxis

        print('mean', centroid, '\naxis', axis, sep='\n')
        return centroid, axis
        # Centroid = np.asmatrix([260.6765,  418.9962,  126.6164])
        # Axis = np.asmatrix(
        # [[0.999957259960338, 2.426512344112703e-04, 0.009242267266263],
        # [0.008617504541717, -0.386586756527017, -0.922224100117558],
        # [0.003349159310558, 0.922253013439533, -0.386545326783471]]
        # )

    def example_relative_position_based_on_refer_point(self):
        pMane_Eye_disp = np.asmatrix(
            [[943.3565, -54.7895, 731.0518],  # "ABS1_P1"
             [727.7784, -48.9640, 40.1491],  # "ABS1_P2"
             [450.7456, -53.8988, 128.5115],  # "ABS1_P3"
             [638.8006, -60.2539, 827.9783],  # "ABS1_P4"
             [1206.5884, -52.3844, 845.1411],  # "ABS2_P1"
             [869.5210, -44.0835, -206.6740],  # "ABS2_P2"
             [-25.2612, -59.8706, 56.7044],  # "ABS2_P3"
             [308.0794, -68.3821, 1105.3155],  # "ABS2_P4"
             [413.1723, -323.2010, 748.0892],  # "DISP_P1"
             [308.8122, -320.1825, 417.1731],  # "DISP_P2"
             [189.7155, -320.8003, 455.8135],  # "DISP_P3"
             [240.2499, -322.3313, 631.7363],  # "DISP_P4"
             [293.9572, -323.7673, 785.1393],  # "DISP_P5"
             [479.4774, -1096.2672, 505.8886],  # "EYE_L",
             [497.3375, -1097.3506, 562.5359],  # "EYE_R",
             [400.4670, -1144.7768, 683.2921],  # "MANE_P1"
             [330.2323, -1142.8242, 457.1621],  # "MANE_P2"
             [194.9668, -1145.1624, 499.9952],  # "MANE_P3"
             [214.4585, -1206.3375, 619.2865],  # "MANE_P4"
             [268.6216, -1147.1164, 725.2041]]  # "MANE_P5"
        )
        # print(pMane_Eye_disp[4:8]) #ABS2
        # print(pMane_Eye_disp[8:13]) #DISP
        # print(pMane_Eye_disp[13:15]) #EYE
        # print(pMane_Eye_disp[16:19]) #MANE_2_3_4

        pPattern_disp = np.asmatrix(
            [[ 94.1548,  -136.3431,  -117.7738],#"DISP_P1",
            [100.1195,  -141.4383,  -464.7347], #"DISP_P2",
            [ -4.8633,  -209.6786,  -464.4936],#"DISP_P3",
            [-12.0288,  -209.4645,  -281.5658], #"DISP_P4",
            [-10.4274,  -204.4850,  -119.0935], #"DISP_P5",
            [ 74.1477,  -148.8633,  -156.4257], #"PATTERN_P1"
            [ 78.1896,  -153.0675,  -426.6260], #"PATTERN_P2"
            [ 4.4950,  -201.0521,  -426.9864],  #"PATTERN_P3"
            [ 0.4278,  -196.8570,  -156.7813],  #"PATTERN_P4"
            [39.3048,  -174.8807,  -291.7078]]  #"PATTERN_P5"
        )
        # print(pPattern_disp[0:5]) #DISP
        # print(pPattern_disp[5:10]) #PATTERN

        p550N_name= np.asmatrix(
            [[ 575.5138,   -53.6211,   347.2357    ],  #"ABS_P1",
            [1296.9180,   -45.9604,  -489.1645    ],  #"ABS_P2",
            [ 605.0637,   -43.4795, -1115.0599    ],  #"ABS_P3",
            [-115.5342,   -50.8216,  -283.4880    ],  #"ABS_P4",
            [  237.6775,  -311.2039,  -388.146455 ],  #"DISP_P1",
            [  465.7575,  -308.9572,  -649.6110   ],  #"DISP_P2",
            [  370.7342,  -307.0648,  -731.1023   ],  #"DISP_P3",
            [  246.9093,  -308.1309,  -596.3134   ],  #"DISP_P4",
            [  143.9568,  -309.2506,  -470.5816   ],  #"DISP_P5",
            [  349.9752,  -903.3901,  -519.6430   ],  #"MANE_P2",
            [  242.8371,  -903.0786,  -612.6906   ],  #"MANE_P3",
            [  150.3462,  -963.7766,  -534.5120  ]]   #"MANE_P4"
        )
        # print(p550N_name[0:4]) #ABS
        # print(p550N_name[4:9]) #DISP
        # print(p550N_name[9:12]) #MANE_2_3_4
                                # MANE_2_3_4      #MANE_2_3_4            #EYE
        p550N_Eye, rr, tt = self.m_findTransformedPoints(pMane_Eye_disp[16:19], p550N_name[9:12], pMane_Eye_disp[13:15] )
        print('p550N_Eye\n', p550N_Eye)
                                                #DISP                #DISP            #PATTERN
        p550N_Pattern, rr, tt = self.m_findTransformedPoints(pPattern_disp[0:5] , p550N_name[4:9], pPattern_disp[5:10] )
        print('p550N_Pattern\n', p550N_Pattern)

        self.m_ProjectDispCoor(p550N_Eye, p550N_Pattern)
        self.m_ProjectDispCoor(np.mean(p550N_Eye, axis=0), p550N_Pattern)
        return

    def example_relative_position_based_on_many_points(self):
        print("relative_position_based_on_many_points\n")
        # m_MakeAbsAxis(1)
        # m_findTransformedPoints(1,1,1)
        # rigid_transform_3D(1,1)
        # m_ProjectAbsCoor(1,1)
        # m_ProjectDispCoor(1,1)
        p660P_Full = np.asmatrix(
        [[1046.714,	-510.129,	-1106.304   ],
        [279.674,	2.586,		200.757         ],
        [757.788,	-93.557,	-665.898        ],
        [1083.243,	-95.258,	-356.687        ],
        [381.708,	-99.336,	-132.29         ],
        [1199.754,	-628.631,	-976.484    ],
        [532.552,	-93.484,	-862.412        ],
        [424.541,	2.715,		304.001         ],
        [1013.188,	-622.468,	-1136.629   ],
        [635.433,	-99.613,	95.958          ],
        [-0.148	,	-0.918,		-0.113              ],
        [172.003,	-0.157,		-175.397        ],
        [170.573,	-1.146,		-1.093          ],
        [346.334,	-0.113,		-0.086          ],
        [169.74	,	-0.87,		175.04              ],
        [1148.348,	-21.663,	-460.659        ],
        [676.332,	-26.053,	-1093.847       ],
        [488.76	,	-402.101,	438.949         ],
        [-115.618,	-52.152,	-283.089        ],
        [575.677,	-53.408,	347.341         ],
        [1296.735,	-44.493,	-489.345        ],
        [604.62	,	-43.492,	-1114.973           ],
        [238.087,	-311.84,	-387.855],
        [466.058,	-309.238,	-649.434],
        [370.987,	-307.535,	-730.875],
        [247.226,	-308.814,	-596.039],
        [144.315,	-310.087,	-470.248],
        [536.218,	-1011.754,	-732.754],
        [429.053,	-1011.639,	-825.74],
        [336.705,	-1072.507,	-747.525]])

        p550N_Full = np.asmatrix(
        [[1046.042,	-511.207,	-1106.107      ],
        [279.699,	2.93,	200.531            ],
        [757.938,	-93.867,	-665.943           ],
        [1083.275,	-96.31,	-356.602           ],
        [381.63,	-99.014,	-132.495               ],
        [1198.847,	-629.936,	-976.362       ],
        [532.802,	-93.418,	-862.55            ],
        [424.51,	2.775,	303.84                 ],
        [1012.32,	-623.404,	-1136.485      ],
        [635.274,	-99.82,	95.876             ],
        [-0.043,	0.052,	-0.453                 ],
        [172.168,	0.587,	-175.655           ],
        [170.666,	-0.437,	-1.355             ],
        [346.42,	0.152,	-0.284                 ],
        [169.762,	-0.278,	174.764            ],
        [1148.593,	-22.802,	-460.521           ],
        [676.842,	-26.108,	-1093.914          ],
        [487.873,	-402.049,	438.767        ],
        [-115.534,	-50.822,	-283.488           ],
        [575.514,	-53.621,	347.236            ],
        [1296.918,	-45.96,	-489.164           ],
        [605.064,	-43.479,	-1115.06           ],
        [237.678,	-311.204,	-388.146       ],
        [465.757,	-308.957,	-649.611       ],
        [370.734,	-307.065,	-731.102       ],
        [246.909,	-308.131,	-596.313       ],
        [143.957,	-309.251,	-470.582       ],
        [349.975,	-903.39,	-519.643],
        [242.837,	-903.079,	-612.691],
        [150.346,	-963.777,	-534.512]])



        p660P = np.asmatrix(
            [[-0.148,	-0.918,	-0.113],
            [172.003,	-0.157,	-175.397],
            [170.573,	-1.146,	-1.093],
            [346.334,	-0.113,	-0.086],
            [169.74,	-0.87,	175.04]]
            )
        p550N = np.asmatrix(
            [[ -0.0430,     0.0524,    -0.4531],
            [172.1681,     0.5869,  -175.6547],
            [170.6656,    -0.4372,    -1.3547],
            [346.4195,     0.1522,    -0.2835],
            [169.7624,    -0.2780,   174.7639]]
            )

        p660P_base501 = np.asmatrix(
            [[0,	    0,	    0],
            [172.151,	0.761,	-175.284],
            [170.721,	-0.228,	-0.98],
            [346.482,	0.805,	0.027],
            [169.888,	0.048,	175.153]]
            )

        p660P_base503 = np.asmatrix(
            [[-170.721,	0.228,	0.98],
            [1.43,	0.989,	-174.304],
            [0,	    0,	    0],
            [175.761,	1.033,	1.007],
            [-0.833,	0.276,	176.133]]
            )
        # print(m_ProjectAbsCoor(p660P,p660P))
        # print(m_ProjectAbsCoor(p550N,p550N))
        primeP2, rr, tt = self.m_findTransformedPoints(p660P,p550N,p660P_Full)
        primeP, rr, tt = self.m_findTransformedPoints(p550N,p660P,p550N_Full)

        print('\n\n')
        print('p550N_Full - primeP2', *(p550N_Full - primeP2), sep='\n')
        print('p660P_Full - primeP', *(p660P_Full - primeP), sep='\n')
        print('\n\n')
        #
        # primeP3, rr, tt = m_findTransformedPoints(p660P, p660P_base501, p660P)
        #
        # primeP4, rr, tt = m_findTransformedPoints(p660P, p660P_base503, p660P_Full)

        print("기준점 계산")
        P_DPA_Pts = np.asmatrix([[391.139000000000, - 1399.38500000000,    706.558000000000],
        [318.651000000000, - 1399.80100000000,    481.105000000000],
        [183.801000000000, - 1399.43800000000,    525.289000000000],
        [203.573000000000, - 1459.04200000000,    645.357000000000],
        [259.687000000000, - 1399.01000000000,    749.782000000000]])

        P_DPA_Ref1 = np.asmatrix([[949.852000000000, - 51.6980000000000,    731.117000000000],
        [727.499000000000, - 53.3340000000000,    42.3240000000000],
        [451.291000000000, - 53.0250000000000,    133.457000000000],
        [646.216000000000, - 51.4270000000000,    831.055000000000]])



        # relative_position_based_on_refer_point()
        #

        # m_ProjectAbsCoor(P_DPA_Pts, P_DPA_Ref1)
        print("********p660P_Full",*p660P_Full[10:15], sep='\n')
        ret1 = self.m_ProjectAbsCoor(p660P_Full, p660P)
        # m_ProjectAbsCoor(p660P_Full-np.asmatrix(np.ones((p660P_Full.shape[0],1))*[170.573,-1.146,	-1.093]), p660P_base503)
        ret2 = self.m_ProjectAbsCoor(p550N_Full, p550N)

        print('ret1', ret1)
        print('ret2', ret2)
        print('Sub1', *(ret1 - ret2), sep='\n')

        ret4 = self.m_ProjectAbsCoor(p660P_Full, p660P_Full[18:22])
        ret5 = self.m_ProjectAbsCoor(p550N_Full, p550N_Full[18:22])

        print('ret4', *ret4, sep='\n' )
        print('ret5', *ret5, sep='\n')
        print('Sub2', *(ret4 - ret5), sep='\n')

        #recovery original position
        ret6, rr, tt = self.m_findTransformedPoints(ret4[18:22], p660P_Full[18:22], ret4)
        ret7, rr, tt = self.m_findTransformedPoints(ret5[18:22], p550N_Full[18:22], ret5)

        print('ret6', *ret6, sep='\n' )
        print('ret7', *ret7, sep='\n')
        print('Sub3', *(ret6 - ret7), sep='\n')

    def example_move_origin_from_refer_point(self):
        print("move_origin_from_refer_point\n")
        p660P_Full = np.asmatrix(
        [[1046.714,	-510.129,	-1106.304   ],
        [279.674,	2.586,		200.757         ],
        [757.788,	-93.557,	-665.898        ],
        [1083.243,	-95.258,	-356.687        ],
        [381.708,	-99.336,	-132.29         ],
        [1199.754,	-628.631,	-976.484    ],
        [532.552,	-93.484,	-862.412        ],
        [424.541,	2.715,		304.001         ],
        [1013.188,	-622.468,	-1136.629   ],
        [635.433,	-99.613,	95.958          ],
        [-0.148	,	-0.918,		-0.113              ],
        [172.003,	-0.157,		-175.397        ],
        [170.573,	-1.146,		-1.093          ],
        [346.334,	-0.113,		-0.086          ],
        [169.74	,	-0.87,		175.04              ],
        [1148.348,	-21.663,	-460.659        ],
        [676.332,	-26.053,	-1093.847       ],
        [488.76	,	-402.101,	438.949         ],
        [-115.618,	-52.152,	-283.089        ],
        [575.677,	-53.408,	347.341         ],
        [1296.735,	-44.493,	-489.345        ],
        [604.62	,	-43.492,	-1114.973           ],
        [238.087,	-311.84,	-387.855],
        [466.058,	-309.238,	-649.434],
        [370.987,	-307.535,	-730.875],
        [247.226,	-308.814,	-596.039],
        [144.315,	-310.087,	-470.248],
        [536.218,	-1011.754,	-732.754],
        [429.053,	-1011.639,	-825.74],
        [336.705,	-1072.507,	-747.525]])

        p660P = np.asmatrix(
            [[-0.148,	-0.918,	-0.113],
            [172.003,	-0.157,	-175.397],
            [170.573,	-1.146,	-1.093],
            [346.334,	-0.113,	-0.086],
            [169.74,	-0.87,	175.04]]
            )
        p660P_4P = np.asmatrix(
            [[-0.148,	-0.918,	-0.113],
            [172.003,	-0.157,	-175.397],
            # [170.573,	-1.146,	-1.093],
            [346.334,	-0.113,	-0.086],
            [169.74,	-0.87,	175.04]]
            )

        p660P_to_origin = np.asmatrix(
            [[-170,	 0,	  0],
            [ 0,	 0,  -170],
            [ 0,	 0,   0],
            [170,	 0,	  0],
            [0,	     0,  170]]
            )

        pTest = np.asmatrix(
            [
            [-3, 0, 0],
            [ 0, 0,-3],
            [ 0, 0, 0],
            [ 0, 0,+3],
            [ 3, 0, 0],

            [0,2,2],
            [2,2,0],
            [2.5,0.5,2.5],
            [2,0,4],
            [4,0,2],

            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            ]
        )
        test_axis = np.asmatrix(
            [[-0.99875234,  0., - 0.04993762],
             [0., - 1., - 0.],
             [-0.04993762, - 0.,    0.99875234]]
        )



        # print(*p660P_Full,'\n\n', *p660P, sep='\n' )
        # print(p660P.shape)
        # print('mean', *np.mean(p660P_4P, axis = 0), sep='\n')
        # print('point - mean\n', *(p660P - np.mean(p660P_4P, axis=0) ), sep='\n')
        # - np.asmatrix([-1.376154e+02, 0.000000e+00, 7.140000e-02])
        print('mean', *np.mean(pTest[0:5], axis = 0), sep='\n')
        print('point - mean\n', *(pTest - np.mean(pTest[0:5], axis=0) ), sep='\n')

        # ret, rr, tt = m_findTransformedPoints(pTest[0:5], p660P_to_origin, pTest)
        # print('final_ret', *ret, sep='\n')
        retData, t_mean2, test_axis2 = self.m_ProjectUserCoor(pTest, pTest[5:10])

        print('retData', retData)
        # m_ProjectUserCoor(p660P_Full, p660P)
        # minus90 = np.asmatrix([[1.57], [1.57], [1.57]]) - np.asmatrix(cv2.Rodrigues(test_axis2)[0])
        # print('t', minus90)
        # print('min90', minus90 * radianToDegree, cv2.Rodrigues(test_axis2)[0] * radianToDegree)
        # print(cv2.Rodrigues(minus90)[0])
        # print(t_mean2)
        print("transform\n", (pTest - np.asmatrix(t_mean2)) * test_axis2 )
        # print("transform\n", (pTest - np.asmatrix(t_mean2)) * (cv2.Rodrigues(minus90)[0]))


        #reference 기준으로 도출된 좌표를 다시 원점 좌표로 복구
        print('test',(test_axis2 *retData[5:10].T).T +  np.asmatrix(t_mean2))
        # print('1',(test_axis2 *pTest[5:10].T).T)
        # print('2',np.asmatrix(t_mean2))
        t_matrix = np.eye(4)
        t_matrix[0:3,0:3] =  test_axis2
        t_matrix[0:3,3] = t_mean2
        print(t_matrix)
        t_matrix_inv = np.linalg.inv(t_matrix)
        print(t_matrix_inv)
        #원점 좌표를 reference 좌표축으로 변환(이 축이 기준이됨 0,0,0) - 역변환
        print('inverse',(t_matrix_inv[0:3,0:3] * pTest[5:10].T).T + t_matrix_inv[0:3,3])

    def example_check_face_pos_GT_based_on_MRA2_CAD_displaycenter(self): #미완성
        print("//////////{:s}//////////".format(sys._getframe().f_code.co_name))
        # GT -> display center -> Camera position -> CAD display center -> modified GT
        nGT = np.asmatrix(
        [[548.0606, -0.9230 ,195],
        [548.1446, -125.9743 ,195],
        [548.1358, 123.9789 ,195],
        [658.2693, -0.8927 ,195],
        [658.2218, -157.4343 ,195],
        [658.4669, 155.4454 ,195],
        [768.5166, -0.9942 ,195],
        [768.3877, -187.8320 ,195],
        [768.5165, 186.0766 ,195],
        [998.9362, -0.9115 ,195],
        [998.4603, -223.7147 ,195],
        [998.7845, 222.0334 ,195]]
        )

        #extract GT position based on display center axis
        pMane_Eye_disp = np.asmatrix(
            [[943.3565, -54.7895, 731.0518],  # "ABS1_P1"
             [727.7784, -48.9640, 40.1491],  # "ABS1_P2"
             [450.7456, -53.8988, 128.5115],  # "ABS1_P3"
             [638.8006, -60.2539, 827.9783],  # "ABS1_P4"
             [1206.5884, -52.3844, 845.1411],  # "ABS2_P1"
             [869.5210, -44.0835, -206.6740],  # "ABS2_P2"
             [-25.2612, -59.8706, 56.7044],  # "ABS2_P3"
             [308.0794, -68.3821, 1105.3155],  # "ABS2_P4"
             [413.1723, -323.2010, 748.0892],  # "DISP_P1"
             [308.8122, -320.1825, 417.1731],  # "DISP_P2"
             [189.7155, -320.8003, 455.8135],  # "DISP_P3"
             [240.2499, -322.3313, 631.7363],  # "DISP_P4"
             [293.9572, -323.7673, 785.1393],  # "DISP_P5"
             [479.4774, -1096.2672, 505.8886],  # "EYE_L",
             [497.3375, -1097.3506, 562.5359],  # "EYE_R",
             [400.4670, -1144.7768, 683.2921],  # "MANE_P1"
             [330.2323, -1142.8242, 457.1621],  # "MANE_P2"
             [194.9668, -1145.1624, 499.9952],  # "MANE_P3"
             [214.4585, -1206.3375, 619.2865],  # "MANE_P4"
             [268.6216, -1147.1164, 725.2041]]  # "MANE_P5"
        )
        # print(pMane_Eye_disp[4:8]) #ABS2
        # print(pMane_Eye_disp[8:13]) #DISP
        # print(pMane_Eye_disp[13:15]) #EYE
        # print(pMane_Eye_disp[16:19]) #MANE_2_3_4

        pPattern_disp = np.asmatrix(
            [[ 94.1548,  -136.3431,  -117.7738],#"DISP_P1",
            [100.1195,  -141.4383,  -464.7347], #"DISP_P2",
            [ -4.8633,  -209.6786,  -464.4936],#"DISP_P3",
            [-12.0288,  -209.4645,  -281.5658], #"DISP_P4",
            [-10.4274,  -204.4850,  -119.0935], #"DISP_P5",
            [ 74.1477,  -148.8633,  -156.4257], #"PATTERN_P1"
            [ 78.1896,  -153.0675,  -426.6260], #"PATTERN_P2"
            [ 4.4950,  -201.0521,  -426.9864],  #"PATTERN_P3"
            [ 0.4278,  -196.8570,  -156.7813],  #"PATTERN_P4"
            [39.3048,  -174.8807,  -291.7078]]  #"PATTERN_P5"
        )
        # print(pPattern_disp[0:5]) #DISP
        # print(pPattern_disp[5:10]) #PATTERN

        p550N_name= np.asmatrix(
            [[ 575.5138,   -53.6211,   347.2357    ],  #"ABS_P1",
            [1296.9180,   -45.9604,  -489.1645    ],  #"ABS_P2",
            [ 605.0637,   -43.4795, -1115.0599    ],  #"ABS_P3",
            [-115.5342,   -50.8216,  -283.4880    ],  #"ABS_P4",
            [  237.6775,  -311.2039,  -388.146455 ],  #"DISP_P1",
            [  465.7575,  -308.9572,  -649.6110   ],  #"DISP_P2",
            [  370.7342,  -307.0648,  -731.1023   ],  #"DISP_P3",
            [  246.9093,  -308.1309,  -596.3134   ],  #"DISP_P4",
            [  143.9568,  -309.2506,  -470.5816   ],  #"DISP_P5",
            [  349.9752,  -903.3901,  -519.6430   ],  #"MANE_P2",
            [  242.8371,  -903.0786,  -612.6906   ],  #"MANE_P3",
            [  150.3462,  -963.7766,  -534.5120  ]]   #"MANE_P4"
        )
        # print(p550N_name[0:4]) #ABS
        # print(p550N_name[4:9]) #DISP
        # print(p550N_name[9:12]) #MANE_2_3_4
                                # MANE_2_3_4      #MANE_2_3_4            #EYE
        p550N_Eye, rr, tt = self.m_findTransformedPoints(pMane_Eye_disp[16:19], p550N_name[9:12], pMane_Eye_disp[13:15] )
        print('p550N_Eye\n', p550N_Eye)
                                                #DISP                #DISP            #PATTERN
        p550N_Pattern, rr, tt = self.m_findTransformedPoints(pPattern_disp[0:5] , p550N_name[4:9], pPattern_disp[5:10] )
        print('p550N_Pattern\n', p550N_Pattern)

        p550N_merge = np.concatenate((p550N_name, p550N_Pattern, p550N_Eye, np.mean(p550N_Eye, axis=0) ))

        pDCenter, Rc, Tc = self.m_ProjectDispCoor(p550N_merge, p550N_Pattern)
        print('pDCenter', pDCenter)
        # m_ProjectDispCoor(np.mean(p550N_Eye, axis=0), p550N_Pattern)

        pDCenter_merge = np.concatenate((pDCenter, nGT, [[0 ,0, 0]], [[1 ,0, 0]], [[11.56351826, 46.22008096, 52.38411943]]))
        print('pDCenter_merge', pDCenter_merge)

        #camera to display center
        T_sample_778569 = np.asmatrix([-13.253501, - 46.178298, - 52.019384])
        R_sample_778569 = np.array([0.070953,    1.948159,    0.105582]) * degreeToRadian
        # R_sample_778569_2 = np.asmatrix([-12.670593, - 45.960195, - 51.981639])
        # T_sample_778569_2 = np.asmatrix([0.071625,    1.971692,     0.083147]) * degreeToRadian
        T_CAD = np.asmatrix([0.00978, -0.04584, -0.04719])
        R_CAD = np.asmatrix([0.0    ,   3.0   ,    0.0]) * degreeToRadian

        # print(cv2.Rodrigues(R_sample_778569)[0])
        # rest = cv2.Rodrigues(p550N_name[0:3])[0]
        print('camera', *((cv2.Rodrigues(R_sample_778569)[0] * pDCenter_merge.T).T + np.asmatrix(T_sample_778569)), sep='\n')
        # print('camera_inv', *((pDCenter_merge - np.asmatrix(T_sample_778569)) * cv2.Rodrigues(R_sample_778569)[0]), sep='\n')

        print('camera_2', *( (pDCenter_merge - np.asmatrix(T_CAD)) * cv2.Rodrigues(R_CAD)[0]), sep='\n')
        # print('camera_2', *((pDCenter_merge - np.asmatrix(T_CAD)) * cv2.Rodrigues(R_CAD)[0]), sep='\n')

        # print(pDCenter_merge, 'ttt\n', T_sample_778569, )

        # t_matrix = np.eye(4)
        # t_matrix[0:3,0:3] =  cv2.Rodrigues(R_sample_778569)[0]
        # t_matrix[0:3,3] = T_sample_778569
        # print(t_matrix)
        # t_matrix_inv = np.linalg.inv(t_matrix)
        # print(t_matrix_inv)
        # #원점 좌표를 reference 좌표축으로 변환(이 축이 기준이됨 0,0,0) - 역변환
        # print('inverse',(t_matrix_inv[0:3,0:3] * pDCenter_merge.T).T + t_matrix_inv[0:3,3])
        return

    def example_test(self):
        mst_739106_01_NG = np.asmatrix(
            [[0.08195928, 0.10489032, 0.69267075],
            [0.17728222, 0.10925468, 0.69065354],
             [0.08181136, 0.10193214, 0.65129581],
             [0.17674347, 0.10588553, 0.64905204],
             [0.08011092, 0.10417334, 0.65523652],
             [0.17484469, 0.10845659, 0.65260795],
             [0, 0, 0]
             ]
        )
        mst_739106_02_NG  = np.asmatrix(
            [[0.07992614, 0.10851276, 0.69048120],
            [0.17564883, 0.11238760, 0.69422094],
             [0.08002846, 0.10547352, 0.65082413],
             [0.17480817, 0.10970294, 0.64958008],
             [0.07812445, 0.10771254, 0.65515808],
             [0.17242126, 0.11228639, 0.65107764],
             [0, 0, 0]
             ]
        )

        mst_739106_03_NG = np.asmatrix(
            [[0.08080904, 0.08298502, 0.69747400],
            [0.17585388, 0.08760310, 0.69544796],
             [0.08076227, 0.08173648, 0.65997013],
             [0.17549017, 0.08614120, 0.65676111],
             [0.07912509, 0.08359874, 0.66174847],
             [ 0.17389188, 0.08795844, 0.65849488],
             [0, 0, 0]
             ]
        )

        mst_739066_01_NG = np.asmatrix(
            [[0.07987873, 0.12114454, 0.68806497],
            [0.17597210, 0.12185083, 0.68512255],
             [0.07997913, 0.11740157, 0.65030805],
             [0.17579403, 0.11818989, 0.64548738],
             [0.07804393, 0.11961369, 0.65077830],
             [0.17341946, 0.12057583, 0.64491026],
             [0, 0, 0]
             ]
        )
        mst_739066_02_NG = np.asmatrix(
            [[0.08167158, 0.09028894, 0.69517437],
            [0.17759340, 0.09120966, 0.69090970],
             [0.08152529, 0.08846443, 0.65793707],
             [0.17716760, 0.08928283, 0.65241108],
             [0.07982656, 0.09032799, 0.65921978],
             [0.17548190, 0.09116799, 0.65374381],
             [0, 0, 0]
             ]
        )
        mst_739095_01_OK = np.asmatrix(
            [[-0.08283047, -0.09724362, 0.68876704],
            [-0.17487075, -0.09416086, 0.70340010],
             [-0.08267753, -0.09506578, 0.65098081],
             [-0.17479649, -0.09216708, 0.66213383],
             [-0.08099957, -0.09703880, 0.65281725],
             [-0.17300063, -0.09427484, 0.66429502],
             [0, 0, 0]
             ]
        )
        mst_739095_02_OK = np.asmatrix(
            [[-0.08212027, -0.08952541, 0.69123739],
            [-0.17442948, -0.08626552, 0.70464047],
             [-0.08194251, -0.08768754, 0.65392189],
             [-0.17418345, -0.08479127, 0.66479405],
             [-0.08021447, -0.08956021, 0.65473711],
             [-0.17246269, -0.08668734, 0.66593508],
             [0, 0, 0]
             ]
        )

        mst_739096_01_OK = np.asmatrix(
            [[-0.08426255, -0.07545503, 0.69987444],
            [-0.17141911, -0.08221066, 0.70400003],
             [-0.08400071, -0.07449988, 0.66304987],
             [-0.17131946, -0.08106609, 0.66452362],
             [-0.08245707, -0.07633561, 0.66460892],
             [-0.16976085, -0.08291501, 0.66613901],
             [0, 0, 0]
             ]
        )
        mst_739096_02_OK = np.asmatrix(
            [[-0.08446433, -0.07516811, 0.69967741],
            [-0.17160085, -0.08198157, 0.70340807],
             [-0.08421073, -0.07420659, 0.66258267],
             [-0.17150571, -0.08077033, 0.66397673],
             [-0.08251467, -0.07604997, 0.66360551],
             [-0.17007068, -0.08263942, 0.66689727],
             [0, 0, 0]
             ]
        )

        print("동일한 세트에서 하나의 켈을 기준으로 R,T값을 추출함")
        print("mst_739106_03_NG 기준")
        # rrrrr, rr, tt= m_findTransformedPoints(mst_739106_03_NG[0:5],mst_739106_02_NG[0:5],mst_739106_03_NG)
        tR, tT = self.rigid_transform_3D(mst_739106_03_NG, mst_739106_02_NG)
        print('tR33', *tR, 'tT(mm)', *tT, sep='\n')
        print('R31(deg)', cv2.Rodrigues(tR)[0] * radianToDegree)
        # print('rrrrr\n', rrrrr)

        tR2, tT2 = self.rigid_transform_3D(mst_739106_03_NG, mst_739106_01_NG)
        print('tR33', *tR2, 'tT(mm)', *tT2, sep='\n')
        print('R31(deg)', cv2.Rodrigues(tR2)[0] * radianToDegree)

        print("mst_739066_02_NG 기준")
        tR, tT = self.rigid_transform_3D(mst_739066_02_NG, mst_739066_01_NG)
        print('tR33', *tR, 'tT(mm)', *tT, sep='\n')
        print('R31(deg)', cv2.Rodrigues(tR)[0] * radianToDegree)

        print("mst_739095_02_OK 기준")
        tR, tT = self.rigid_transform_3D(mst_739095_02_OK, mst_739095_01_OK)
        print('tR33', *tR, 'tT(mm)', *tT, sep='\n')
        print('R31(deg)', cv2.Rodrigues(tR)[0] * radianToDegree)

        print("mst_739096_02_OK 기준")
        tR, tT = self.rigid_transform_3D(mst_739096_02_OK, mst_739096_01_OK)
        print('tR33', *tR, 'tT(mm)', *tT, sep='\n')
        print('R31(deg)', cv2.Rodrigues(tR)[0] * radianToDegree)

class RecoveryCtrl():
    def __del__(self):
        print("*************delete RecoveryCtrl class***********\n")
    def __init__(self):
        self.debugflag = C_PRINT_CTRL_ENABLE
        self.objCore = RecoveryAlgo()
        print("*************initialize RecoveryCtrl class***********\n")
        self.progress_pos = 0
        self.progress_end = 100

    def load_DPA_file(self, fname):
        """
        Load a DPA file and return the data as a pandas DataFrame.

        Parameters:
            fname (str): The file name of the DPA file to be loaded.

        Returns:
            pandas.DataFrame: The data loaded from the DPA file.
        """
        print("//////////", funcname(), "//////////")

        df = pd.read_csv(fname, skipinitialspace = True, header=None)
        # print(df.head())
        df.columns = ['title', 'number', 'point_name', 'tx', 'ty', 'tz']
        df = df.sort_values(['title','point_name','number'], ascending=(True,True,True))
        # print(tData)
        return df

    def save_DPA_file(self, tdatas, filename):
        """
        A function to save the given DataFrame to an Excel file.
        
        Args:
            tdatas (DataFrame): The DataFrame to be saved.
            filename (str): The name of the file to which the DataFrame should be saved.
        """
        print("//////////", funcname(), "//////////")
        if(tdatas.empty is True):
            print("저장할 데이터가 없습니다.")
            return
        tdata = tdatas.copy()
        tdata.point_name = '"' + tdata.point_name + '"'

        fname, fext = os.path.splitext(filename)
        # print(fname, fext)
        # tdata.to_csv(fname+".ext", mode='w', index=False, header=False, sep=',', quotechar=" ", float_format='%.4f')
        print(fname+".xls")
        # print(tdata)

        # 엑셀파일 만들고 열기.
        wb = Workbook()
        ws = wb.active

        for r in dataframe_to_rows(tdata, index=True, header=True):
            ws.append(r)

        try:
            wb.save(fname + ".xlsx")
            # tdata.to_excel(fname + ".xlsx")  # xls저장
        except:  # <- naked except is a bad idea
            tdata.to_csv(fname+".ext", mode='w', index=False, header=False, sep=',', quotechar=" ", float_format='%.4f')
            messagebox.showerror("Save Data File", "Failed to save file\n'%s'" % (fname + ".xls"))

        if(tdata.get(['group_sub']) is not None):
            del tdata['group_sub']
            # tdata = tdata.drop(['group_sub'], axis=1)
            print("is group_sub")
        if(tdata.get(['group_first']) is not None):
            del tdata['group_first']
            # tdata = tdata.drop(['group_first'], axis=1)
            print("is group_first")
        if(tdata.get(['seq']) is not None):
            del tdata['seq']
            # tdata = tdata.drop(['seq'], axis=1)
            print("is seq")

        tdata.to_csv(filename, mode='w', index=False, header=False, sep=',', quotechar=" ", float_format='%.4f')

        # print(tdata)
        print(filename)
        pass


    #첫번째 DataFrame에서 두번째 DataFrame을 비교하여, point_name의 값이 없는 부분을 모두 (0,0,0)으로 생성함
    def compare_between_title(self, tfirst, tcomp):
        """
        A function to compare between two titles and perform specific operations on them.
        
        Parameters:
        tfirst : DataFrame
            The first title to compare.
        tcomp : DataFrame
            The title to compare against the first title.
        
        Returns:
        DataFrame
            A DataFrame containing the merged and sorted data from the input titles.
        """
        print("//////////{:s}//////////".format(sys._getframe().f_code.co_name))
        tdebug = 0
        tfirst = tfirst.reset_index(drop=True)
        if(tdebug):
            print('tfirst',tfirst)
        # print('tcompare',tcompare)
        tcompare = tcomp.copy()
        tcompare['title'] = tfirst.title[0]
        # tcompare['number'] += 1000
        tcompare['number'] =  tcompare.apply(lambda x: ( x['number'] + 10000* x['group_first']) if x['number']< 9999 else (x['number'] + (10 ** (self.digit_length(x['number']))) * x['group_first']), axis = 1)
        tcompare['tx'] = 0.0
        tcompare['ty'] = 0.0
        tcompare['tz'] = 0.0
        tcompare['group_first'] = tfirst.group_first[0]
        tcompare = tcompare.reset_index(drop=True)
        df = tcompare

        for i in tfirst.point_name:
            for tnum, j in enumerate(tcompare.point_name):
                if(i == j):
                    df = df.drop(tnum)
        # print(tcompare)
        if (tdebug):
            print('df', df)
            print('tfirst',tfirst)
        # ret = tfirst.merge([df])
        ret = pd.concat([tfirst,df])
        ret = ret.sort_values(['title', 'point_name', 'number'], ascending=(True, True, True))
        ret = ret.reset_index(drop=True)
        if (tdebug):
            print('ret', ret)
        return ret

    #중복데이터 제거 (argument1, argument2) / return argument2.drop.duplicate
    def check_duplicate(self, tdata_one, tdata_two):
        """
        A function to check for duplicates between two dataframes and update one of them accordingly.
        
        Parameters:
            self: reference to the current instance of the class
            tdata_one: the first dataframe to compare
            tdata_two: the second dataframe to compare
        
        Returns:
            A modified dataframe after removing duplicates
        """
        print("//////////{:s}//////////".format(sys._getframe().f_code.co_name))
        tdebug = 0
        # ttext = 'DEL'
        tdata_copy = tdata_two.copy()
        for i, ttwo in tdata_copy.iterrows():
            # if (tdebug):
            #   print('i tx,ty,tz', i, tone['tx'], tone['ty'], tone['tz'])
            for j, tone in tdata_one.iterrows():
                if((tone['title'] != ttwo['title']) or(tone['group_sub'] != ttwo['group_sub'])):
                    continue

                if(tone['tx'] == ttwo['tx'] and tone['ty'] == ttwo['ty'] and tone['tz'] == ttwo['tz']):
                    tdata_copy = tdata_copy.drop(i, axis=0)
                elif( abs(tone['tx'] - ttwo['tx']) <= 1 and abs(tone['ty'] - ttwo['ty']) <= 1 and abs(tone['tz'] - ttwo['tz']) <= 1 ):
                    tdata_copy = tdata_copy.drop(i, axis=0)
                elif((tone['title'] == ttwo['title']) and (tone['point_name'].split("|")[0] == ttwo['point_name'].split("|")[0])):
                    tdata_copy['point_name'][i] = str(ttwo['point_name']) + "|" + str(ttwo['group_first'])

        tdata_copy = tdata_copy.reset_index(drop=True)
        if (tdebug):
            print('tdata_two',tdata_two)
            print('tdata_copy',tdata_copy)
        return tdata_copy

    def check_duplicate_and_remove(self, tdatas):
        """
        A function that checks for duplicates in the input data, removes them, and returns the cleaned data.
        """
        print("//////////{:s}//////////".format(sys._getframe().f_code.co_name))
        tdebug = 0

        tdata_copy = tdatas.copy()

        # df_title = tdata_copy.groupby('title').count().index
        # df_type = tdata_copy[['group_sub', 'title']].drop_duplicates()
        # print('df_type',df_type.count())
        # df5_list = df3[~df3['group_sub'].str.contains("\*")].reset_index(drop=True)
        df_sort = tdata_copy.sort_values(['title', 'group_sub', 'point_name'],ascending=(True, True, True)).reset_index(drop=True)
        df_sort['compare_group'] = df_sort['point_name'].str.split('|').str[0]
        print('df_sort', len(df_sort), df_sort)

        df_type = df_sort[['title', 'compare_group']].drop_duplicates()
        # print('df_type',df_type)
        # print('df_type',df_type.values)

        print('checking')
        getData = pd.DataFrame()
        for tTitle, tGcomp in df_type.values:
            print(tTitle, tGcomp)
            df_temp = df_sort[(df_sort['title'] == tTitle) & (df_sort['compare_group'] == tGcomp)]
            if (tdebug):
                print('df_temp', df_temp)

            for i, tleft in df_temp.iterrows():
                for j, tright in df_temp.iterrows():
                    if(i >= j):
                        continue
                    # if (abs(tleft['tx'] - tright['tx']) <= C_MAX_GAP and abs(tleft['ty'] - tright['ty']) <= C_MAX_GAP and abs(tleft['tz'] - tright['tz']) <= C_MAX_GAP):
                    if (abs(tleft['tx'] - tright['tx']) <= C_MAX_GAP ):
                        if(abs(tleft['ty'] - tright['ty']) <= C_MAX_GAP and abs(tleft['tz'] - tright['tz']) <= C_MAX_GAP):
                            print('삭제', j,'번째', i)
                            df_temp = df_temp.drop(j, axis=0)
                            # continue
            getData = pd.concat([getData, df_temp])

        del (getData['compare_group'])
        if (tdebug):
            print('getData', len(getData), getData)

        return getData


    #숫자 자릿수 리턴
    def digit_length(self, n):
        """
        Calculate the length of a number by counting the number of digits. 

        Parameters:
            n (int): The number to calculate the length of.

        Returns:
            int: The length of the number.
        """
        ans = 0
        while n:
            n //= 10
            ans += 1
        return ans

    def check_available_regid_transform(self, tfirst, tsecond, tfirst_rest):
        """
        A function to check the availability of certain transformations based on input data.

        Parameters:
            self: The object instance.
            tfirst: The first set of data for transformation.
            tsecond: The second set of data for transformation.
            tfirst_rest: The rest of the first set of data for transformation.

        Returns:
            checkOK: A boolean indicating if the transformation is available.
            tleft: The transformation matrix of the first set of data.
            tright: The transformation matrix of the second set of data.
        """
        checkOK = True
        if (len(tfirst_rest) == 0):
            checkOK = False
        if(len(tfirst) != len(tsecond)):
            checkOK = False
        if (len(tfirst) < C_MIN_VALUE_RIGID_CALC or len(tsecond) < C_MIN_VALUE_RIGID_CALC):
            checkOK = False
            #두 인풋의 갯수가 다를때, 같은 것이 있는지 체크
            # tfirst['point_name'] == tsecond['point_name']
            # tModifiedFirst = tfirst[tfirst['point_name'] == tsecond['point_name']]
            # print(tModifiedFirst)
        tleft = np.asmatrix(tfirst[['tx','ty','tz']])
        tright = np.asmatrix(tsecond[['tx','ty','tz']])
        return checkOK, tleft, tright

    def decrypt_divide_same_type(self, ttype, tfirst, tsecond, tdatas):
        """
        A function to decrypt, divide, and sort data based on specific types and titles.
        Parameters:
            - ttype: The type to filter the data with.
            - tfirst: The first title to process.
            - tsecond: The second title to process.
            - tdatas: The data to be decrypted and sorted.
        Returns:
            A DataFrame containing the processed and sorted data.
        """
        print("//////////{:s}//////////".format(sys._getframe().f_code.co_name), ttype, tfirst, tsecond)
        tdebug = 1
        tdata2 = tdatas.copy()

        tdata_first_type = tdata2[(tdata2['group_sub'] == ttype) & (tdata2['title'] == tfirst) ].sort_values(['number'], ascending=True).reset_index(drop=True)
        tdata_first_without = tdata2[(tdata2['group_sub'] != ttype) & (tdata2['title'] == tfirst)].sort_values(['number'], ascending=True).reset_index(drop=True)

        tdata_second_type = tdata2[(tdata2['group_sub'] == ttype) & (tdata2['title'] == tsecond) ].sort_values(['number'], ascending=True).reset_index(drop=True)
        tdata_second_without = tdata2[(tdata2['group_sub'] != ttype) & (tdata2['title'] == tsecond)].sort_values(['number'], ascending=True).reset_index(drop=True)

        tdata_rest_title = tdata2[(tdata2['title'] != tfirst) & (tdata2['title'] != tsecond)].sort_values(['number'], ascending=True).reset_index(drop=True)

        tdata_first_type['type_idx'] = tdata_first_type.point_name.str.split('|').str[1:].str.join(sep='|')
        tdata_first_type = tdata_first_type.sort_values(['title', 'type_idx', 'point_name'], ascending=True)
        tcount1_type = pd.Series(tdata_first_type['type_idx'] ).value_counts()
        tUnit1Cnt = pd.Series(tdata_first_type['type_idx'] ).value_counts().values[0]
        tdata_first_type = tdata_first_type[~tdata_first_type['type_idx'].isin(tdata_first_type['type_idx'].value_counts()[tdata_first_type['type_idx'].value_counts() != tUnit1Cnt].index)]
        tType1Cnt = len(pd.Series(tdata_first_type['type_idx'] ).value_counts())
        del(tdata_first_type['type_idx'])

        tdata_second_type['type_idx'] = tdata_second_type.point_name.str.split('|').str[1:].str.join(sep='|')
        tdata_second_type = tdata_second_type.sort_values(['title', 'type_idx', 'point_name'], ascending=True)
        tcount2_type = pd.Series(tdata_second_type['type_idx'] ).value_counts()
        tUnit2Cnt = pd.Series(tdata_second_type['type_idx'] ).value_counts().values[0]
        tdata_second_type = tdata_second_type[~tdata_second_type['type_idx'].isin(tdata_second_type['type_idx'].value_counts()[tdata_second_type['type_idx'].value_counts() != tUnit2Cnt].index)]
        tType2Cnt = len(pd.Series(tdata_second_type['type_idx'] ).value_counts())
        #하나의 title에서 동일 type에대해 갯수가 다른것에 한해, 제거한다
        del(tdata_second_type['type_idx'])

        if(tUnit1Cnt < C_MIN_VALUE_RIGID_CALC or tUnit2Cnt < C_MIN_VALUE_RIGID_CALC):
            print("Skip tUnitCnt are",tUnit1Cnt, tUnit2Cnt)
            return False, tdatas

        # for i1 in range(0, tUnit1Cnt * tType1Cnt, tUnit1Cnt):
        #     tempBase = tdata_first_type[i1: i1 + tUnit1Cnt]
        #     tempBase = tempBase.sort_values(['point_name'], ascending=True)
        #     print('tdata_base', tempBase, "\n")
        #
        # for i2 in range(0, tUnit2Cnt * tType2Cnt , tUnit2Cnt):
        #     tempBase = tdata_second_type[i2: i2 + tUnit2Cnt]
        #     tempBase = tempBase.sort_values(['point_name'], ascending=True)
        #     print('tdata_base', tempBase, "\n")

        print("\n****************")
        totalDf = pd.DataFrame()
        tCnt = 0
        for i1 in range(0, tUnit1Cnt * tType1Cnt, tUnit1Cnt):
            tempBase1 = tdata_first_type[i1: i1 + tUnit1Cnt]
            # tempBase1 = tempBase1.sort_values(['point_name'], ascending=True).reset_index(drop=True)
            tempBase1 = tempBase1.reset_index(drop=True)
            for i2 in range(0, tUnit2Cnt * tType2Cnt, tUnit2Cnt):
                tCnt +=1
                tempBase2 = tdata_second_type[i2: i2 + tUnit2Cnt]
                # tempBase2 = tempBase2.sort_values(['point_name'], ascending=True).reset_index(drop=True)
                tempBase2 = tempBase2.reset_index(drop=True)
                print('tCnt', tCnt)
                print('tdata_base1', tempBase1, "\n")
                print('tdata_base2', tempBase2, "\n")
                ret = True
                ret2 = False
                # if(tUnit1Cnt != tUnit2Cnt):
                #     ret, tempBaseOne, tempBaseTwo = update_unitType_position_using_rigid(tempBase1, tempBase2)
                # else:
                #     tempBaseOne = tempBase1.copy()
                #     tempBaseTwo = tempBase2.copy()
                ret, tempBaseOne, tempBaseTwo = self.update_unitType_position_using_rigid(tempBase1, tempBase2)
                # if(ret == True):
                ret2, tempBaseOneRest, tempBaseTwoRest = self.update_OtherType_position_using_rigid(ttype, tempBaseOne, tempBaseTwo, tdata_first_without, tdata_second_without, tCnt)
                totalDf = pd.concat([totalDf, tempBaseOne, tempBaseTwo, tempBaseOneRest, tempBaseTwoRest,])

            print('\n')
        # totalDf['type_idx'] = totalDf.point_name.str.split('|').str[1:].str.join(sep='|')
        # totalDf = totalDf.sort_values(['title','type_idx', 'point_name'], ascending=True)
        # totalDf = totalDf.drop_duplicates(['title','point_name'], keep='first').reset_index(drop=True)
        # del(totalDf['type_idx'])
        # totalDf[['group_first']] = totalDf.groupby('title').grouper.group_info[0] + 1

        totalDf['type_idx'] = totalDf.point_name.str.split('|').str[1:].str.join(sep='|')
        totalDf = totalDf.sort_values(['title', 'type_idx', 'point_name'], ascending=True)
        totalDf = totalDf.drop_duplicates(['title', 'point_name'], keep='first').reset_index(drop=True)
        del (totalDf['type_idx'])
        totalDf[['group_first']] = (totalDf.groupby('title').grouper.group_info[0] + 1)[0]  #.reshape(totalDf.groupby('title').grouper.group_info[0].size,1)
        print('totalDf', totalDf)

        retTotalDf = self.check_duplicate_and_remove(totalDf)
        retTotalDf = pd.concat([retTotalDf, tdata_rest_title])
        retTotalDf['type_idx'] = retTotalDf.point_name.str.split('|').str[1:].str.join(sep='|')
        retTotalDf = retTotalDf.sort_values(['title', 'type_idx', 'point_name'], ascending=True).reset_index(drop=True)
        del (retTotalDf['type_idx'])
        retTotalDf[['group_first']] = (retTotalDf.groupby('title').grouper.group_info[0] + 1)[0]
        print('retTotalDf', len(retTotalDf), retTotalDf)

        return retTotalDf

    def update_OtherType_position_using_rigid(self, ttype, tfirst, tsecond, trest_first, trest_second, tCount):
        """
        A function that updates the position of a certain type using rigid calculations.

        Parameters:
        ttype (str): The type to be updated.
        tfirst (DataFrame): The first set of data.
        tsecond (DataFrame): The second set of data.
        trest_first (DataFrame): The remaining first data.
        trest_second (DataFrame): The remaining second data.
        tCount (int): The count parameter.

        Returns:
        bool: Indicates if the update was successful or not.
        DataFrame: The updated first type data.
        DataFrame: The updated second type data.
        """
        print("//////////{:s}//////////".format(sys._getframe().f_code.co_name))
        tdebug = 1

        if(len(tfirst) < C_MIN_VALUE_RIGID_CALC or len(tsecond) < C_MIN_VALUE_RIGID_CALC):
            return False, trest_first, trest_second
        # tdata_first_type2['point_name'].str.split('|').str[0]
        # PATTERN_P1
        # tdata_first_type2['point_name'].str.split('|').str[:3].str.join(sep='-')
        #PATTERN_P1-1-3
        # tdata_first_type2['point_name'].str.split('|').str[1:].str.join(sep='-')
        #1-3

        tdata_type_first_dup2 = (tfirst.copy()).reset_index(drop=True)
        tdata_type_second_dup2 = (tsecond.copy()).reset_index(drop=True)
        tdata_type_first_rest2 = (trest_first.copy()).reset_index(drop=True)
        tdata_type_second_rest2 = (trest_second.copy()).reset_index(drop=True)

        tdata_type_first_dup = np.asmatrix(tdata_type_first_dup2[['tx', 'ty', 'tz']])
        tdata_type_second_dup = np.asmatrix(tdata_type_second_dup2[['tx', 'ty', 'tz']])
        tdata_type_first_rest = np.asmatrix(tdata_type_first_rest2[['tx', 'ty', 'tz']])
        tdata_type_second_rest = np.asmatrix(tdata_type_second_rest2[['tx', 'ty', 'tz']])

        print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        print(tdata_type_first_dup, '\n', tdata_type_first_rest, '\n\n', tdata_type_second_dup, '\n',  tdata_type_second_rest)

        tdata_first_type3 = trest_first.copy()
        tdata_second_type3 = trest_second.copy()
        if(len(tdata_type_first_rest) > 0 and len(tdata_type_first_dup) >= C_MIN_VALUE_RIGID_CALC):
            update_type_rest_data, tR_1_to_2, tT_1_to_2 = self.objCore.m_findTransformedPoints(tdata_type_first_dup, tdata_type_second_dup,
                                                                                  tdata_type_first_rest)
            update_type_rest = tdata_type_first_rest2.copy()
                    # tdata2[((tdata2['group_sub'] == ttype) & (tdata2['title'] == tfirst)) & ~(tdata2['point_name'].isin(tdata_second_type2['point_name']))].reset_index(drop=True)
            # update_type_rest = tdata2[(tdata2['group_sub'] == ttype) & ~(tdata2['point_name'].isin(tdata_second_type2['point_name']))].reset_index(drop=True)

            update_type_rest[['tx', 'ty', 'tz']] = pd.DataFrame(update_type_rest_data, columns=['tx', 'ty', 'tz'])[['tx', 'ty', 'tz']]
            # update_type_rest['title'] = tsecond
            update_type_rest['title'] = tdata_type_second_dup2['title'][0]
            update_type_rest['number'] = update_type_rest.apply(lambda x: (x['number'] + 10000 * x['group_first']) if x['number'] < 9999 else (x['number'] + (10 ** (digit_length(x['number']))) * x['group_first']), axis=1)
            update_type_rest['seq'] = update_type_rest['seq'] + ">" + ttype + ' ' + update_type_rest['group_first'].astype(str) +"_" + str(tCount)

            # if(tdata_type_second_dup2.point_name.str.split('|').str[1:].str.join(sep='|').any() != ""):
            #     # update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0] + '|' + tdata_type_second_dup2['point_name'].str.split('|').str[1:].str.join(sep='|')
            #     update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0] + '|' + tdata_type_second_dup2['point_name'].str.split('|').str[1:].str.join(sep='|') + '|' + update_type_rest['group_first'].astype(str)
            # else:
            #     # update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0]
            #     update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0] + '|' + update_type_rest['group_first'].astype(str)
            update_type_rest['point_name'] = update_type_rest['point_name'] + '|' + update_type_rest['group_first'].astype(str) +"_" + str(tCount)

            # 중복데이터 제거
            # tdata_first_without_type = check_duplicate(tdata2, tdata_first_without_type)

            if (tdebug):
                print('\nrecovery large point in type second', update_type_rest)
            tdata_second_type3 = pd.concat([tdata_second_type3, update_type_rest])

        if (len(tdata_type_second_rest) > 0 and len(tdata_type_first_dup) >= C_MIN_VALUE_RIGID_CALC):
            update_type_rest_data, tR_2_to_1, tT_2_to_1 = self.objCore.m_findTransformedPoints(tdata_type_second_dup,
                                                                                  tdata_type_first_dup,
                                                                                  tdata_type_second_rest)
            update_type_rest = tdata_type_second_rest2.copy()

            update_type_rest[['tx', 'ty', 'tz']] = pd.DataFrame(update_type_rest_data, columns=['tx', 'ty', 'tz'])[['tx', 'ty', 'tz']]
            update_type_rest['title'] = tdata_type_first_dup2['title'][0]
            update_type_rest['number'] = update_type_rest.apply(lambda x: (x['number'] + 10000 * x['group_first']) if x['number'] < 9999 else (x['number'] + (10 ** (digit_length(x['number']))) * x['group_first']), axis=1)
            update_type_rest['seq'] = update_type_rest['seq'] + ">" + ttype + ' ' + update_type_rest['group_first'].astype(str) +"_" + str(tCount)
            # if (tdata_type_first_dup2.point_name.str.split('|').str[1:].str.join(sep='|').any() != ""):
            #     # update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0] + '|' + tdata_type_first_dup2['point_name'].str.split('|').str[1:].str.join(sep='|')
            #     update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0] + '|' + tdata_type_first_dup2['point_name'].str.split('|').str[1:].str.join(sep='|') + '|' + update_type_rest['group_first'].astype(str)
            # else:
            #     # update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0]
            #     update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0] + '|' + update_type_rest['group_first'].astype(str)
            update_type_rest['point_name'] = update_type_rest['point_name'] + '|' + update_type_rest['group_first'].astype(str) +"_" + str(tCount)

            # 중복데이터 제거
            # tdata_first_without_type = check_duplicate(tdata2, tdata_first_without_type)
            if (tdebug):
                print('\nrecovery large point in type first', update_type_rest)
            tdata_first_type3 = pd.concat([tdata_first_type3, update_type_rest])

        return True, tdata_first_type3, tdata_second_type3


    def update_unitType_position_using_rigid(self, tdata_first_type2, tdata_second_type2):
        """
        A function to update unit type position using a rigid transformation.

        Parameters:
        - tdata_first_type2: The first type of data for the transformation
        - tdata_second_type2: The second type of data for the transformation

        Returns:
        - A tuple containing a boolean indicating success, the updated first type data, and the updated second type data
        """
        print("//////////{:s}//////////".format(sys._getframe().f_code.co_name))
        tdebug = 1

        if(len(tdata_first_type2) < C_MIN_VALUE_RIGID_CALC or len(tdata_second_type2) < C_MIN_VALUE_RIGID_CALC):
            return False, tdata_first_type2, tdata_second_type2
        # tdata_first_type2['point_name'].str.split('|').str[0]
        # PATTERN_P1
        # tdata_first_type2['point_name'].str.split('|').str[:3].str.join(sep='-')
        #PATTERN_P1-1-3
        # tdata_first_type2['point_name'].str.split('|').str[1:].str.join(sep='-')
        #1-3

        #first type과 second type의 갯수가 같은지 보고, 다르면, 같은 것과 다른것을 구분한다.
        # 구분뒤에 다른것을 복원하여, tdata2에 업데이트 한다.
        #마지막으로 tdata_first_type과 tdata_second_type을 복원한다
        # tdata_type_first_dup2 = tdata_first_type2[(tdata_first_type2['point_name'].isin(tdata_second_type2['point_name'])) ].reset_index(drop=True)
        # tdata_type_second_dup2 = tdata_second_type2[(tdata_second_type2['point_name'].isin(tdata_first_type2['point_name'])) ].reset_index(drop=True)
        # tdata_type_first_rest2 = tdata_first_type2[~(tdata_first_type2['point_name'].isin(tdata_second_type2['point_name'])) ].reset_index(drop=True)
        # tdata_type_second_rest2 = tdata_second_type2[~(tdata_second_type2['point_name'].isin(tdata_first_type2['point_name'])) ].reset_index(drop=True)
        tdata_type_first_dup2 = tdata_first_type2[(tdata_first_type2['point_name'].str.split('|').str[0].isin(tdata_second_type2['point_name'].str.split('|').str[0]))].reset_index(drop=True)
        tdata_type_second_dup2 = tdata_second_type2[(tdata_second_type2['point_name'].str.split('|').str[0].isin(tdata_first_type2['point_name'].str.split('|').str[0]))].reset_index(drop=True)
        tdata_type_first_rest2 = tdata_first_type2[~(tdata_first_type2['point_name'].str.split('|').str[0].isin(tdata_second_type2['point_name'].str.split('|').str[0]))].reset_index(drop=True)
        tdata_type_second_rest2 = tdata_second_type2[~(tdata_second_type2['point_name'].str.split('|').str[0].isin(tdata_first_type2['point_name'].str.split('|').str[0]))].reset_index(drop=True)

        tdata_type_first_dup = np.asmatrix(tdata_type_first_dup2[['tx', 'ty', 'tz']])
        tdata_type_second_dup = np.asmatrix(tdata_type_second_dup2[['tx', 'ty', 'tz']])
        tdata_type_first_rest = np.asmatrix(tdata_type_first_rest2[['tx', 'ty', 'tz']])
        tdata_type_second_rest = np.asmatrix(tdata_type_second_rest2[['tx', 'ty', 'tz']])

        print("llllllllllllllllllllllllllllll")
        print(tdata_type_first_dup, '\n',tdata_type_first_rest, '\n\n', tdata_type_second_dup,'\n', tdata_type_second_rest)

        tdata_first_type3 = tdata_first_type2.copy()
        tdata_second_type3 = tdata_second_type2.copy()
        if(len(tdata_type_first_rest) > 0 and len(tdata_type_first_dup) >= C_MIN_VALUE_RIGID_CALC):
            update_type_rest_data, tR_1_to_2, tT_1_to_2 = self.objCore.m_findTransformedPoints(tdata_type_first_dup, tdata_type_second_dup,
                                                                                  tdata_type_first_rest)
            update_type_rest = tdata_type_first_rest2.copy()
                    # tdata2[((tdata2['group_sub'] == ttype) & (tdata2['title'] == tfirst)) & ~(tdata2['point_name'].isin(tdata_second_type2['point_name']))].reset_index(drop=True)
            # update_type_rest = tdata2[(tdata2['group_sub'] == ttype) & ~(tdata2['point_name'].isin(tdata_second_type2['point_name']))].reset_index(drop=True)

            update_type_rest[['tx', 'ty', 'tz']] = pd.DataFrame(update_type_rest_data, columns=['tx', 'ty', 'tz'])[['tx', 'ty', 'tz']]
            # update_type_rest['title'] = tsecond
            update_type_rest['title'] = tdata_type_second_dup2['title'][0]
            update_type_rest['number'] = update_type_rest.apply(lambda x: (x['number'] % 10000 + (tdata_type_second_dup2['number']//10000)*10000) , axis=1)
            # update_type_rest['number'] = update_type_rest['number'] + (tdata_type_second_dup2['number']//10000)*10000
            if(tdata_type_second_dup2.point_name.str.split('|').str[1:].str.join(sep='|').any() != ""):
                update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0] + '|' + tdata_type_second_dup2['point_name'].str.split('|').str[1:].str.join(sep='|')
            else:
                update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0]
            # update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0] + tdata_type_second_dup2['point_name'].str.split('|').str[1:].str.join(sep='-')
            #
            # update_type_rest.apply(
            #     lambda x: (x if ((x = tdata_type_second_dup2['point_name'].str.split('|').str[1:].str.join(sep='-')) == "") else ("|" + x), axis=1))
            update_type_rest['seq'] = update_type_rest['seq'].astype(str) + "(" + update_type_rest['group_first'].astype(str) + ")"

            if (tdebug):
                print('recovery small point in type second', update_type_rest)
            tdata_second_type3 = pd.concat([tdata_second_type2, update_type_rest])
            tdata_second_type3 = tdata_second_type3.sort_values(['point_name'], ascending=True).reset_index(drop=True)

        if (len(tdata_type_second_rest) > 0 and len(tdata_type_first_dup) >= C_MIN_VALUE_RIGID_CALC):
            update_type_rest_data, tR_2_to_1, tT_2_to_1 = self.objCore.m_findTransformedPoints(tdata_type_second_dup,
                                                                                  tdata_type_first_dup,
                                                                                  tdata_type_second_rest)
            update_type_rest = tdata_type_second_rest2.copy()

            update_type_rest[['tx', 'ty', 'tz']] = pd.DataFrame(update_type_rest_data, columns=['tx', 'ty', 'tz'])[
                ['tx', 'ty', 'tz']]
            update_type_rest['title'] = tdata_type_first_dup2['title'][0]
            update_type_rest['number'] = update_type_rest.apply(lambda x: (x['number'] % 10000 +(tdata_type_first_dup2['number'] // 10000) * 10000), axis=1)
            # update_type_rest['number'] = update_type_rest['number'] + (tdata_type_first_dup2['number'] // 10000) * 10000
            if (tdata_type_first_dup2.point_name.str.split('|').str[1:].str.join(sep='|').any() != ""):
                update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0] + '|' + \
                                                 tdata_type_first_dup2['point_name'].str.split('|').str[1:].str.join(sep='|')
            else:
                update_type_rest['point_name'] = update_type_rest['point_name'].str.split('|').str[0]
            update_type_rest['seq'] = update_type_rest['seq'].astype(str) + "(" + update_type_rest['group_first'].astype(str) + ")"

            if (tdebug):
                print('recovery small point in type first', update_type_rest)
            tdata_first_type3 = pd.concat([tdata_first_type2, update_type_rest])
            tdata_first_type3 = tdata_first_type3.sort_values(['point_name'], ascending=True).reset_index(drop=True)

        return True, tdata_first_type3, tdata_second_type3


    def update_position_using_relative_2title(self, ttype, tfirst, tsecond, tdatas):
        """
        A function that updates position using relative to title with the given parameters.
        
        Parameters:
            ttype (type): The type of the title.
            tfirst (first): The first title.
            tsecond (second): The second title.
            tdatas (datas): The data to be processed.
        
        Returns:
            retData: The processed data after decryption and division.
        """
        print("//////////", funcname(), "//////////")
        # print('tdatas',tdatas)
        retData = self.decrypt_divide_same_type(ttype, tfirst, tsecond, tdatas)
        return retData

    def auto_recovery_3d_points_on_each_of_coordinate(self, tDatas):
        print("//////////",funcname(),"//////////")
        # print("//////////{:s}//////////".format(funcname()))
        tData = tDatas.copy()
        tData['group_first'] = (tData.groupby('title').grouper.group_info[0] + 1)[0]
        tData['group_sub'] = tData['point_name'].str.split('_').str[0]
        tData['seq'] = ""
        # print(tData)
        print(tData.head())
        print(tData.tail())

        print("**" * 50)
            # Labeling이 동일한 Title 추출 (기준점을 찾기위함)
        # [title, group_sub] 데이터중에 중복된 데이터 삭제
        df3 = tData[['group_sub', 'title']].drop_duplicates()
        df5_list = df3[~df3['group_sub'].str.contains("\*")].reset_index(drop=True)
        print('\ndf5_list\n',df5_list)

        # [title] 기준으로 (중복 제거) group_sub의 분류된 label 갯수
        df7_title = df5_list.groupby('title').count() \
            .sort_values(['group_sub'], ascending=False)
        print('\ndf7_title\n',df7_title)

        # [group_sub] 기준으로 (중복 제거) title의 분류된 label 갯수
        df9_group_sub = df5_list.groupby('group_sub').count() \
            .sort_values(['title'], ascending=False)
        print('\ndf9_group_sub\n',df9_group_sub)

        tvalidData = tData[~tData['point_name'].str.contains('\*')].reset_index(drop=True)
        print('tvalidData', tvalidData)
        tmodifiedData = tvalidData

        # 첫번째 좌표묶음기준으로 나머지 좌표들의 point_name이 없는 부분을 모두 생성하여, 더미 (0,0,0)을 생성함
        tfirst_title = pd.DataFrame()
        for tnum, (tkey, tdata) in enumerate(tvalidData.groupby(['title'])):
            # print('key', tnum, tkey,tdata )
            if(tnum == 0):
                tfirst_title = tdata
            else:
                tfirst_title = self.compare_between_title(tfirst_title, tdata)

        # print('tfirst_title', tfirst_title)

        # 첫번째 모두 생성하여, 더미 (0,0,0)을 기준으로 나머지 좌표묶음들의 더미(0,0,0) 생성함
        tdata_first = tfirst_title.copy()
        tsecond_title_all = pd.DataFrame()
        tsecond_title = pd.DataFrame()
        for tnum, (tkey, tdata) in enumerate(tvalidData.groupby(['title'])):
            # print('key', tnum, tkey,tdata )
            if(tnum == 0):
                continue
            else:
                tsecond_title = tdata
                tsecond_title_all = pd.concat([tsecond_title_all, self.compare_between_title(tsecond_title, tdata_first)])

        ttitle_all = pd.concat([tfirst_title, tsecond_title_all]).reset_index(drop=True)
        # print('tfirst_title', tfirst_title)
        print('ttitle_all', ttitle_all)

        df8_list_all = ttitle_all[['group_sub', 'title']].drop_duplicates().reset_index(drop=True)
        print('\ndf8_list_all\n',df8_list_all)

        print("\n[group_sub] 기준으로 (중복 제거) title의 분류된 label의 갯수가 2개이상인 데이터 추출\n")
        tData_grp= []
        for tnum in range(0,len(df5_list.group_sub.value_counts().index),1):
            # print(tnum)
            ta = int(df5_list.group_sub.value_counts()[tnum])
            if(ta > 1 ):
                tb = df5_list.group_sub.value_counts().index[tnum]
                # print(ta, tb)
                tData_grp.append([ta, tb])
        print('tData_grp', tData_grp)

        df5_list = df5_list.sort_values(['group_sub', 'title'], ascending=(True, True)).reset_index(drop=True)
        df5_list_dup = df5_list.copy()
        # print('df5_list',df5_list)
        for i, row in df5_list.iterrows():
            # print(i, row['group_sub'], row['title'])
            bchk = 1
            for jcount, jtype in tData_grp:
                if(jtype == row['group_sub']):
                    bchk = 0
                    break
            if(bchk == 1):
                df5_list_dup = df5_list_dup.drop(i)

        df5_list_dup = df5_list_dup.reset_index(drop=True)
        print('\ndf5_list_dup', df5_list_dup)

        for i, (jcount, jtype) in enumerate(tData_grp):
            print(jtype, '->', list(df5_list_dup['title'][df5_list_dup['group_sub'] == jtype]))
            tData_grp[i].append(list(df5_list_dup['title'][df5_list_dup['group_sub'] == jtype]))
            # tData_grp[i].append(list(df5_list_dup['title'][df5_list_dup['group_sub'] == jtype]))

        print('\ntData_grp', tData_grp)

        update_list = df8_list_all.copy()
        for i, tdat in df8_list_all.iterrows():
            for j, tdat2 in df5_list_dup.iterrows():
                if(tdat['group_sub'] == tdat2['group_sub'] and tdat['title'] == tdat2['title']):
                    update_list = update_list.drop(i, axis = 0)

        print('\nupdate_list',update_list)

        #같은 point_name의 title을 2가지씩 추출할 조합
        combination_of_title = list(combinations(list(df7_title.index), 2))
        print('combination', )
        for i, (jcount, jtype, jcomp) in enumerate(tData_grp):
            # if(jtype == "MANE"):
            #     continue
            print(i, jcomp)
            # print(list(combinations(jcomp, 2)))
            tloop = list(combinations(jcomp, 2))
            for title_one, title_two in tloop:
                print('\t',jtype , '->' ,title_one, 'vs' ,title_two )
                for j, (j_title_one, j_title_two) in enumerate(combination_of_title):
                    if((j_title_one == title_one and j_title_two == title_two) or (j_title_one == title_two and j_title_two == title_one)):
                        del combination_of_title[j]
                        print('j', j_title_one, j_title_two)
                        break
                tvalidData = self.update_position_using_relative_2title(jtype, title_one, title_two, tvalidData)
        print('combination_of_title',combination_of_title)
        for title_one, title_two in combination_of_title:
            print('\t', jtype, '->', title_one, 'vs', title_two)
            tvalidData = self.update_position_using_relative_2title(jtype, title_one, title_two, tvalidData)
        print('final tvalidData',tvalidData)


        print("\nRRRRRRRRRRRRRR")
        return tvalidData

        for tidx, (tcount, tlabel) in enumerate(tData_grp):
            print(tidx, tcount, tlabel)
            # print(tidx, tdata[0], tdata[1])
            for (tcol_title,tcol_group_sub), tdata in tvalidData.groupby(['title', 'group_sub']):
                print(tcol_title, tcol_group_sub)
                if(tcol_group_sub == tlabel):
                # if(tkey[1] == )
                    print(tlabel, '\n', np.array(tdata[['tx', 'ty', 'tz']].reset_index(drop=True)))

        for tkey, tdata in tvalidData.groupby(['title', 'group_sub']):
            # print(tkey,'\n', tdata)
            print(tkey,'\n', np.array(tdata[['tx','ty','tz']].reset_index(drop=True)))
            # for tdata in tData[~tData['point_name'].str.contains('\*')]:
            # tData[~tData['point_name'].str.contains('\*')].reset_index(drop=True)
            # print(tData[~tData['point_name'].str.contains('\*') and tData['title'] == tname].reset_index(drop=True))
            # print(tdata)

        aaaa = np.asmatrix(
            [[-0.148,	-0.918,	-0.113],
            [172.003,	-0.157,	-175.397],
            [170.573,	-1.146,	-1.093],
            [346.334,	-0.113,	-0.086],
            [169.74,	-0.87,	175.04]

             ]
        )
        bbbb = np.asmatrix(
            [[ -0.0430,     0.0524,    -0.4531],
            [172.1681,     0.5869,  -175.6547],
            [170.6656,    -0.4372,    -1.3547],
            [346.4195,     0.1522,    -0.2835],
            [169.7624,    -0.2780,   174.7639]
             ]
        )

        # p660P = np.asmatrix(
        #     [[-0.148,	-0.918,	-0.113],
        #     [172.003,	-0.157,	-175.397],
        #     [170.573,	-1.146,	-1.093],
        #     [346.334,	-0.113,	-0.086],
        #     [169.74,	-0.87,	175.04]]
        #     )
        # p550N = np.asmatrix(
        #     [[ -0.0430,     0.0524,    -0.4531],
        #     [172.1681,     0.5869,  -175.6547],
        #     [170.6656,    -0.4372,    -1.3547],
        #     [346.4195,     0.1522,    -0.2835],
        #     [169.7624,    -0.2780,   174.7639]]
        #     )

        rrrrr, rr, tt = m_findTransformedPoints(aaaa,bbbb,aaaa)
        print('rrrrr\n', rrrrr)
        tR, tT = rigid_transform_3D(aaaa, bbbb)

        print('tR33', *tR, 'tT(mm)', *tT, sep='\n')

        print('R31(deg)', cv2.Rodrigues(tR)[0] * radianToDegree)

        # print('camera', *((cv2.Rodrigues(tR)[0] * aaaa[0:5].T).T + np.asmatrix(tT)), sep='\n')
        print(aaaa[0:5].T.shape, tT.shape, tR.shape)
        print('1', (tR * aaaa[0:5].T).T )
        print('2', tT.T )
        result = (tR * aaaa.T).T + tT.T
        print('RT -> camera', *result , sep='\n')
        # print('camera', (tR * aaaa[0:5]).T + np.asmatrix(tT), sep='\n')

        print('rollback RT -> camera_2', *((result - tT.T) * tR), sep='\n')

        return



    def preprocess(self, tDatas):
        """
        A function to preprocess the input data by performing various data manipulations and extractions.
        This function takes in a pandas DataFrame 'tDatas' and returns multiple processed DataFrames.
        """
        print("//////////",funcname(),"//////////")
        # print("//////////{:s}//////////".format(funcname()))

        tData = tDatas.copy()
        tData['group_first'] = (tData.groupby('title').grouper.group_info[0] + 1)[0]
        tData['group_sub'] = tData['point_name'].str.split('_').str[0]
        tData['seq'] = ""
        # print(tData)
        # print(tData.head())
        # print(tData.tail())

        print("**" * 50)
            # Labeling이 동일한 Title 추출 (기준점을 찾기위함)
        # [title, group_sub] 데이터중에 중복된 데이터 삭제
        df3 = tData[['group_sub', 'title']].drop_duplicates()
        df5_list = df3[~df3['group_sub'].str.contains("\*")].reset_index(drop=True)
        # print('\ndf5_list\n',df5_list)
        #중복제거를 통한 데이터 출력 기준

        # [title] 기준으로 (중복 제거) group_sub의 분류된 label 갯수
        # df7_title = df5_list.groupby('title').count() \
        #     .sort_values(['group_sub'], ascending=False)
        # print('\ndf7_title\n',df7_title)
        #Title을 기준으로 Sort 뒤에 데이터 영역을 나눌때 필요

        # [group_sub] 기준으로 (중복 제거) title의 분류된 label 갯수
        df9_group_sub = df5_list.groupby('group_sub').count() \
            .sort_values(['title'], ascending=False)
        # print('\ndf9_group_sub\n',df9_group_sub)
        # print('\ndf9_group_sub\n',df9_group_sub[df9_group_sub.values>1])
        df9_group_sub_2_more = df9_group_sub[df9_group_sub.values>1]

        print("\n[group_sub] 기준으로 (중복 제거) title의 분류된 label의 갯수가 2개이상인 데이터 추출\n")
        tData_grp = np.concatenate(
            (np.column_stack(df9_group_sub.index.values).T, np.column_stack(df9_group_sub.title).T), axis=1).tolist()
        if(len(df9_group_sub_2_more)):
            tData_grp2 = np.concatenate(
                (np.column_stack(df9_group_sub_2_more.title).T, np.column_stack(df9_group_sub_2_more.index.values).T),
                axis=1).tolist()
        else:
            tData_grp2 = []

        # print("tData_grp", tData_grp)


        df5_list2 = df5_list.sort_values(['group_sub', 'title']).reset_index(drop=True)
        df5_list_dup = df5_list2.copy()
        # print('df5_list',df5_list)
        for i, row in df5_list2.iterrows():
            # print(i, row['group_sub'], row['title'])
            bchk = 1
            for jcount, jtype in tData_grp2:
                if(jtype == row['group_sub']):
                    bchk = 0
                    break
            if(bchk == 1):
                df5_list_dup = df5_list_dup.drop(i)

        df5_list_dup = df5_list_dup.reset_index(drop=True)
        # print('\ndf5_list_dup', df5_list_dup)

        for i, (jcount, jtype) in enumerate(tData_grp2):
            print(jtype, '->', list(df5_list_dup['title'][df5_list_dup['group_sub'] == jtype]))
            tData_grp2[i].append(list(df5_list_dup['title'][df5_list_dup['group_sub'] == jtype]))
            # tData_grp[i].append(list(df5_list_dup['title'][df5_list_dup['group_sub'] == jtype]))

        print('\ntData_grp2', tData_grp2)

        tvalidData = tData[~tData['point_name'].str.contains('\*')].reset_index(drop=True)
        # print('tvalidData', tvalidData)

        return df5_list, tData_grp, tData_grp2, tvalidData

    def extract_dup_type_between_titles(self, tfirst, tsecond, tdatas , inputlist, tIdx):
        """
        A function to extract duplicate types between titles from the given data and input list.

        :param tfirst: The first title
        :param tsecond: The second title
        :param tdatas: The input data
        :param inputlist: The input list
        :param tIdx: The index
        :return: A tuple containing a boolean indicating if duplicates were found and the duplicate type
        """
        print("//////////",funcname(),"//////////")
        tdebug = 1
        print("extract_dup_type_between_titles", tfirst, tsecond, 'inputlist=',inputlist)
        tdata2 = tdatas.copy()
        ret = False
        retType = ""

        tdata2_dup = tdata2[['group_sub', 'title']][(tdata2['title'] == tfirst) | (tdata2['title'] == tsecond)].drop_duplicates()

        tdata2_dup = tdata2_dup.groupby('group_sub').count() \
            .sort_values(['title'], ascending=False)
        print('tdata2_dup', tdata2_dup)

        tdata2_dup2 = tdata2_dup[tdata2_dup.values>1]
        print('tdata2_dup2', tdata2_dup2.count(), tdata2_dup2)

        # if(tIdx == C_TAB3 or tIdx == C_TAB4):
        #     inputlist = []

        if(tdata2_dup2.count().title >=1):
            if(inputlist == [] or tIdx == C_TAB3 or tIdx == C_TAB4):
                ret = True
                retType = tdata2_dup2.index[0]
                print(ret, retType)
            else:
                if (tIdx == C_TAB1):
                    for icount, iname, icomp in inputlist:
                        for lname in tdata2_dup2.index:
                            if(iname == lname):
                                print("Project Check (type=%s)"%(iname))
                                ret = True
                                retType = lname
                else:
                    for icount, iname, icomp in inputlist:
                        for lname in tdata2_dup2.index:
                            if(iname == lname):
                                print("Project Check (type=%s)({%s},{%s}),({%s},{%s})"%(iname, tfirst,icomp.count(tfirst) ,tsecond, icomp.count(tsecond) ))
                                if(int(icomp.count(tfirst))>=1 and int(icomp.count(tsecond))>=1):
                                    ret = True
                                    retType = lname

        print("match", ret, retType)
        return ret, retType

    def calc_auto_recovery_3d_points(self, tDatas, tIdx, dictData, skipData):
        print("//////////",funcname(),"//////////")
        retFlag = True
        retText = ""
        tdebug = 0
        print("Tab#",tIdx)
        for key, value in dictData.items():
            if value.get():
                print(key)
        print("")
        retC, tIdx, dictData, retData, tSkipData = self.parsing_selected_data_by_tabType(tIdx, dictData, skipData)

        if(retC == False):
            retFlag = retC
            retText = retData
            return retFlag, retText, tDatas

        ##########################################################################
        tData = tDatas.copy()
        title_cnt = tData.groupby('title').title.ngroups
        if(title_cnt >= 2):
            self.progress_end = len(list(combinations(range(0, title_cnt), 2)))
        else:
            self.progress_end = title_cnt
        self.progress_pos = 0


        tData['group_first'] = (tData.groupby('title').grouper.group_info[0] + 1)[0]
        tData['group_sub'] = tData['point_name'].str.split('_').str[0]
        tData['seq'] = ""
        if(tdebug):
            print(tData.head())
            print(tData.tail())
        print('tData_all', len(tData),tData)
        tData_skip = pd.DataFrame()
        for itype in tSkipData:
            tData_skip = pd.concat([tData_skip , tData[tData['group_sub']==itype]])
            tData = tData[~(tData['group_sub']==itype)]
        print('tData_skip', len(tData_skip),tData_skip)
        print('tData', len(tData), tData)

        if(tIdx == C_TAB3):
            tModifiedData = tData.copy() #pd.DataFrame()
            for ititle, itype in retData.tolist():
                tModifiedData = tModifiedData[~((tModifiedData['title'] == ititle) & (tModifiedData['group_sub'] == itype))]
            print(tModifiedData)
            tData = tModifiedData
        elif(tIdx == C_TAB4):
            tModifiedData = tData.copy() #pd.DataFrame()
            for ititle, ipointname, itx, ity, itz  in retData:
                tModifiedData = tModifiedData[~((tModifiedData['title'] == ititle) & (tModifiedData['point_name'] == ipointname) & (tModifiedData['tx'] == float(itx)) & (tModifiedData['ty'] == float(ity)) & (tModifiedData['tz'] == float(itz)) )]
            print(tModifiedData)
            tData = tModifiedData
        elif(tIdx == C_TAB5):
            print("Tab5")
            tType = retData
            return retFlag, tType, tData

        print("**" * 50)
        # [title, group_sub] 데이터중에 중복된 데이터 삭제
        df3 = tData[['group_sub', 'title']].drop_duplicates()
        df5_list = df3[~df3['group_sub'].str.contains("\*")].reset_index(drop=True)
        print('\ndf5_list\n',df5_list)

        print("\n[group_sub] 기준으로 (중복 제거) title의 분류된 label의 갯수가 2개이상인 데이터 추출\n")
        tData_grp= []
        if (tIdx == C_TAB1):
            tData_grp = copy.deepcopy(retData.tolist())
        # elif (tIdx == C_TAB2):
        #     for i in retData.tolist()
        else:
            for tnum in range(0,len(df5_list.group_sub.value_counts().index),1):
                # print(tnum)
                ta = int(df5_list.group_sub.value_counts()[tnum])
                if(ta > 1 ):
                    tb = df5_list.group_sub.value_counts().index[tnum]
                    # print(ta, tb)
                    tData_grp.append([ta, tb])
                    # break
            # [group_sub] 기준으로 (중복 제거) title의 분류된 label 갯수
            # df9_group_sub = df5_list.groupby('group_sub').count() \
            #     .sort_values(['title'], ascending=False)
            # print('\ndf9_group_sub\n', df9_group_sub)
            # df9_group_sub_2_more = df9_group_sub[df9_group_sub.values > 1]

            # tData_grp = np.concatenate(
            #     (np.column_stack(df9_group_sub_2_more.index.values).T, np.column_stack(df9_group_sub_2_more.title).T), axis=1).tolist()

        print('tData_grp', tData_grp)
        tData_grp_add = copy.deepcopy(tData_grp)

        ##################################################

        df5_list_dup = df5_list.copy()
        for i, row in df5_list.iterrows():
            # print(i, row['group_sub'], row['title'])
            bchk = 1
            for jcount, jtype in tData_grp:
                if(jtype == row['group_sub']):
                    bchk = 0
                    break
            if(bchk == 1):
                df5_list_dup = df5_list_dup.drop(i)

        df5_list_dup = df5_list_dup.sort_values(['group_sub', 'title'], ascending=(True, True)).reset_index(drop=True)
        print('\ndf5_list_dup', df5_list_dup)

        if (tIdx == C_TAB2):
            print("2")
            # retData.tolist()
            for i, (jcount, jtype) in enumerate(tData_grp_add):
                tAvailable = []
                lproject = list(df5_list_dup['title'][df5_list_dup['group_sub'] == jtype])
                print(jtype, '->', list(df5_list_dup['title'][df5_list_dup['group_sub'] == jtype]))

                print("Check available according to checked box")
                for ktype, kproject in retData.tolist():
                    for lpro in lproject:
                        if(kproject == lpro and ktype == jtype):
                            tAvailable.append(lpro)
                # if(len(tAvailable) > 0):
                tData_grp_add[i].append(tAvailable)
            print('\ntData_grp_add', tData_grp_add)

        else:
            for i, (jcount, jtype) in enumerate(tData_grp_add):
                print(jtype, '->', list(df5_list_dup['title'][df5_list_dup['group_sub'] == jtype]))
                tData_grp_add[i].append(list(df5_list_dup['title'][df5_list_dup['group_sub'] == jtype]))
            print('\ntData_grp_add', tData_grp_add)

        #################################################################

        tvalidData = tData[~tData['point_name'].str.contains('\*')].reset_index(drop=True)
        print('tvalidData', tvalidData)

        ################################################################

        # [title] 기준으로 (중복 제거) group_sub의 분류된 label 갯수
        df7_title = df5_list.groupby('title').count().sort_values(['group_sub'], ascending=False)
        print('\ndf7_title\n',df7_title)

        #같은 point_name의 title을 2가지씩 추출할 조합
        combination_of_title = list(combinations(list(df7_title.index), 2))
        print('combination', combination_of_title)
        ttcnt = 1
        if (1):
            for i, (jcount, jtype, jcomp) in enumerate(tData_grp_add):
                # if(jtype == "MANE"):
                #     continue
                print(i, jcomp)
                # print(list(combinations(jcomp, 2)))
                tloop = list(combinations(jcomp, 2))
                for title_one, title_two in tloop:
                    print('\t',jtype , '->' ,title_one, 'vs' ,title_two )
                    for j, (j_title_one, j_title_two) in enumerate(combination_of_title):
                        if((j_title_one == title_one and j_title_two == title_two) or (j_title_one == title_two and j_title_two == title_one)):
                            print('j', j_title_one, j_title_two, '- delete')
                            del combination_of_title[j]
                            print(ttcnt, 'combination rest', combination_of_title)
                            ttcnt += 1
                            tvalidData = self.update_position_using_relative_2title(jtype, title_one, title_two, tvalidData)
                            self.progress_pos = self.progress_end - len(combination_of_title)
                            break
        elif (0):
            for i, (jcount, jtype, jcomp) in enumerate(tData_grp_add):
                # if(jtype == "MANE"):
                #     continue
                print(i, jcomp)
                # print(list(combinations(jcomp, 2)))
                tloop = list(combinations(jcomp, 2))
                for title_one, title_two in tloop:
                    print('\t',jtype , '->' ,title_one, 'vs' ,title_two )
                    for j, (j_title_one, j_title_two) in enumerate(combination_of_title):
                        if((j_title_one == title_one and j_title_two == title_two) or (j_title_one == title_two and j_title_two == title_one)):
                            del combination_of_title[j]
                            print('j', j_title_one, j_title_two)
                            break
                    print(ttcnt, 'combination check', combination_of_title)
                    ttcnt+=1
                    tvalidData = self.update_position_using_relative_2title(jtype, title_one, title_two, tvalidData)
                    self.progress_pos = self.progress_end - len(combination_of_title)
        else:
            for title_one, title_two in combination_of_title:
                # print('hehe',list(df5_list['group_sub'][df5_list_dup['title'] == title_one or df5_list_dup['title'] == title_two]))
                ret, jtype = self.extract_dup_type_between_titles(title_one, title_two, tvalidData, tData_grp_add, tIdx )
                if(ret == True):
                    print('\t', jtype, '->', title_one, 'vs', title_two)
                    tvalidData = self.update_position_using_relative_2title(jtype, title_one, title_two, tvalidData)
            #combination_of_title을 모든 경우의 수로 넣지말고, 체크된 포인트그룹 커버하는 수의 조합으로 for문을 돌린다면
            combination_of_title = list(combinations(list(df7_title.index), 2))

        print('combination_of_title', combination_of_title)
        for idx, (title_one, title_two) in enumerate(combination_of_title):
            # print('hehe',list(df5_list['group_sub'][df5_list_dup['title'] == title_one or df5_list_dup['title'] == title_two]))
            ret, jtype = self.extract_dup_type_between_titles(title_one, title_two, tvalidData, tData_grp_add, tIdx )
            if(ret == True):
                print(ttcnt, 'combination last')
                ttcnt += 1
                print('\t', jtype, '->', title_one, 'vs', title_two)
                tvalidData = self.update_position_using_relative_2title(jtype, title_one, title_two, tvalidData)
                self.progress_pos = self.progress_end - (len(combination_of_title) - idx -1 )
        # print('final tvalidData',tvalidData)

        tRet_Merge = pd.concat([tvalidData, tData_skip])
        tRet_Merge['type_idx'] = tRet_Merge.point_name.str.split('|').str[1:].str.join(sep='|')
        tRet_Merge = tRet_Merge.sort_values(['title', 'type_idx', 'point_name'], ascending=(True, True, True))
        del(tRet_Merge['type_idx'])
        print('final tvalidData',tRet_Merge)

        print("\nRRRRRRRRRRRRRR")
        self.progress_pos = self.progress_end

        return retFlag, retText, tRet_Merge

    def calc_relative_position_on_base_type(self, tBaseType, rData):
        """
        Calculate the relative position of the base type based on the provided data.

        Parameters:
        - tBaseType: The base type to calculate the relative position for.
        - rData: The data containing titles and coordinates.

        Returns:
        - retC: A boolean indicating if the calculation was successful.
        - retData: A DataFrame containing the calculated relative positions.
        """
        ttitles = np.asmatrix(rData['title'].drop_duplicates().reset_index(drop=True)).T.tolist()
        print(ttitles)
        self.progress_end = len(ttitles)
        self.progress_pos = 0
        # print(ttitles.shape)
        retC = False
        retData = pd.DataFrame()
        for ipos, ititle in enumerate(ttitles):
            print(ititle[0])
            # tdata_baseType = rData[['point_name', 'tx', 'ty', 'tz']][(rData['group_sub'] == tBaseType[0]) & (rData['title'] == ititle[0]) ]
            tdata_baseType = rData[(rData['group_sub'] == tBaseType[0]) & (rData['title'] == ititle[0])]
            if(len(tdata_baseType) == 0):
                continue
            tdata_baseType['type_idx'] = tdata_baseType.point_name.str.split('|').str[1:].str.join(sep='|')
            tdata_baseType = tdata_baseType.sort_values(['title', 'type_idx', 'point_name'], ascending=(True, True, True))


            tPatternCnt = pd.Series(tdata_baseType['type_idx']).value_counts().values[0]
            print('tPatternCnt', tPatternCnt)
            tdata_baseType = tdata_baseType[~tdata_baseType['type_idx'].isin(tdata_baseType['type_idx'].value_counts()[tdata_baseType['type_idx'].value_counts() != tPatternCnt].index)]

            tdata_all = rData[((rData['group_sub'] != tBaseType[0])) & (rData['title'] == ititle[0])].reset_index(drop=True)
            tdata_all['type_idx'] = tdata_all.point_name.str.split('|').str[1:].str.join(sep='|')

            if(len(tdata_baseType) > 0):
                for i in range(0, len(tdata_baseType), tPatternCnt):
                    tempBase = tdata_baseType[i: i + tPatternCnt]
                    # tempBase = tempBase.sort_values(['point_name'], ascending=True)
                    print('tdata_base', tempBase, "\n")
                    # print('tdata_all', tdata_all)
                    tBase = np.asmatrix(tempBase[['tx', 'ty', 'tz']])
                    tTarget = np.asmatrix(tdata_all[['tx', 'ty', 'tz']])
                    print('\n','tBase',tBase)
                    print('tTarget',tTarget)
                    # m_ProjectDispCoor(p550N_Eye, p550N_Pattern)
                    # m_ProjectDispCoor(np.mean(p550N_Eye, axis=0), p550N_Pattern)
                    retRelativePos, rr, tt = self.objCore.m_ProjectDispCoor(tTarget, tBase)
                    # print(np.round(retRelativePos,4))
                    tempdata_all = tdata_all.copy()
                    tempdata_all[['tx', 'ty', 'tz']] = pd.DataFrame(retRelativePos, columns=['tx', 'ty', 'tz'])[['tx', 'ty', 'tz']]
                    # tempdata_all['point_name'] = tempdata_all['point_name'] + '/' + tempdata_all['type_idx']
                    # if (tempdata_all.type_idx.any() != ""):
                    # if (tempdata_all.point_name.str.split('|').str[1:].str.join(sep='|').any() == []):
                    #     tempdata_all['point_name'] = tempdata_all['point_name']
                    # else:
                    #     tempdata_all['point_name'] = tempdata_all['point_name'] + '/' + tempdata_all['type_idx']

                    # print('tempdata_all', tempdata_all)
                    print("\n")
                    retData = pd.concat([retData, tempdata_all])
                    retC = True

            self.progress_pos = ipos
        # self.check_duplicate_and_remove(retData)

        if(retC==True):
            retData = retData.sort_values(['title', 'type_idx', 'point_name'], ascending=(True, True, True))
            del retData['type_idx']
        # del retData['seq']
        print('\nretData', retData)
        print('EEEEEEEEEEEEEEEEEEEE')
        self.progress_pos = self.progress_end
        return retC, retData

    def parsing_selected_data_by_tabType(self, tIdx, dictData, skipData):
        """
        A function to parse selected data based on the tab type, taking into account the dictionary data and skip data. Returns various data based on the tab type.
        """
        print("//////////",funcname(),"////////// ", tIdx)
        nCnt = 0
        nRet = True
        retData = ""

        tdictData = []
        tdicDataValue = []
        tSkipData = []

        for key, value in dictData.items():
            if value.get():
                nCnt += 1
        if(nCnt == 0):
            retData = "체크박스가 선택되지 않았습니다."
            print(retData)
            return False, tIdx, dictData, retData, tSkipData


        if(tIdx == C_TAB1):
            for key, value in dictData.items():
                if (value.get() and int(key.split('|')[1]) > 1):
                    tdictData.append(key.split('|')[1])
                    tdicDataValue.append(key.split('\t')[0])
            if(tdictData == []):
                retData = "중복개수 2이상인 포인트 그룹이 선택되지 않았으니, 다시 선택하시오!!"
                print(retData)
                return False, tIdx, dictData, retData, tSkipData
            print(tdictData)
            print(tdicDataValue)
            retData = np.concatenate((np.array(tdictData).reshape(-1,1), np.array(tdicDataValue).reshape(-1,1)), axis=1)
            print(retData)
        elif(tIdx == C_TAB2):
            for key, value in dictData.items():
                if value.get():
                    tdictData.append(key.split('\t')[0])
                    tdicDataValue.append(key.split('|')[1])
            print(tdictData)
            print(tdicDataValue)
            for i in tdictData:
                # print(tdictData.count(i),'개')
                if(int(tdictData.count(i))==1):
                    retData = retData + i + " "
                    nRet = False
            if(nRet == False):
                retData = retData + "포인트 그룹이 1개씩 선택된 되었습니다. 포인트 그룹을 2개이상 선택하시오!!"
                print(retData)
                return nRet, tIdx, dictData, retData, tSkipData
            retData = np.concatenate((np.array(tdictData).reshape(-1, 1), np.array(tdicDataValue).reshape(-1, 1)), axis=1)
            print(retData)
        elif (tIdx == C_TAB3):
            for key, value in dictData.items():
                if value.get():
                    tdictData.append(key.split('|')[1])
                    tdicDataValue.append(key.split('\t')[0])
            print(tdictData)
            print(tdicDataValue)
            retData = np.concatenate((np.array(tdictData).reshape(-1, 1), np.array(tdicDataValue).reshape(-1, 1)), axis=1)
            print(retData)
        elif(tIdx == C_TAB4):
            for key, value in dictData.items():
                if value.get():
                    tdictData.append(key.split('\t')[0])
                    tdicDataValue.append([key.split('\t')[0],key.split('!')[1],
                                          key.split('!')[2].split(',')[0],key.split('!')[2].split(',')[1],key.split('!')[2].split(',')[2]])
            print(tdictData)
            print(tdicDataValue)
            retData = tdicDataValue
        elif(tIdx == C_TAB5):
            if(nCnt != 1):
                retData = "1개만 선택해야합니다!!"
                print(retData)
                return False, tIdx, dictData, retData, tSkipData

            for key, value in dictData.items():
                if value.get():
                    tdictData.append(key.split('\t')[0])
            print(tdictData)
            retData = tdictData

        ####################Skip type implimentation
        for key, value in skipData.items():
            if value.get():
                tSkipData.append(key.split(' |')[0])
        # print(tSkipData)

        return nRet, tIdx, dictData, retData, tSkipData

C_MENU_WIDTH = 640
C_MENU_HEIGHT = 720
C_MENU_POS_X = 100
C_MENU_POS_Y = 100
C_MENU_MAINTITLE = "3D Auto recovery position -V1.03"

class mainMenu_GUI():
    def __init__(self):
        self.tt = 0
        self.root = tk.Tk()
        self.root.title(C_MENU_MAINTITLE)
        # self.root.geometry("640x400+100+100")
        self.root.geometry("%dx%d+%d+%d"%(C_MENU_WIDTH,C_MENU_HEIGHT,C_MENU_POS_X,C_MENU_POS_Y  ))
        self.root.resizable(True, True)

        self.label = tk.Label(self.root, text="1. 3D position 데이터를 읽으세요\n2.서치후 중복되는 라벨을 출력합니다. 기준으로 설정하고 싶은 라벨을 체크하세요.")
        self.label.pack()

        self.menubar = tk.Menu(self.root)

        filemenu = tk.Menu(self.menubar, tearoff=0)

        self.menubar.add_cascade(label='File', menu=filemenu)
        filemenu.add_command(label="New", command=self.menu_new)
        filemenu.add_command(label="Load Data", command=self.menu_load)
        filemenu.add_command(label="Save Data", command=self.menu_save)
        filemenu.add_separator()
        filemenu.add_command(label="Save Log", command=self.menu_save_log)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)

        helpmenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About", command=self.help_about)

        self.notebook = tkinter.ttk.Notebook(self.root, width=C_MENU_WIDTH-40, height=C_MENU_HEIGHT-240-80)
        # self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.pack()

        self.label2 = tk.Label(self.root, text="")
        self.label2.pack()

        self.progress = tkinter.ttk.Progressbar(self.root, orient="horizontal", length=C_MENU_WIDTH-20, mode="determinate")
        self.progress.pack()
        # self.progress.grid(row=row + 5, sticky=tk.N)
        # self.progress.grid(row=row + 5, sticky=tk.S, pady=100)


        # self.textlog = tk.Text(self.root, width=86)
        self.textlog = tk.Text(self.root, width=100, font=("Helvetica", 8))
        tscroll = tk.Scrollbar(self.root, command=self.textlog.yview, orient=tk.VERTICAL)
        tscroll.config(command=self.textlog.yview)
        self.textlog.configure(yscrollcommand=tscroll.set)
        self.textlog.pack(side=tk.LEFT, fill=tk.Y, expand='YES')
        tscroll.pack(side=tk.RIGHT, fill=tk.BOTH)


        self.initial_data()
        self.menu_new()

        sys.stdout = Logger(self.textlog)
        self.ObjCtrl = RecoveryCtrl()


    def initial_data(self):
        self.tOne = threading.Thread()
        self.mframe = [0, 0, 0, 0, 0]
        self.mframeIdx = 0

    def alert_msg(self, ttext, tcolor="IndianRed1"):
        if(ttext == ""):
            self.label2.configure(text="", bg='SystemButtonFace' )
        else:
            self.label2.configure(text = str(ttext), bg=tcolor)

    def menu_new(self):
        self.textlog.delete('1.0',tk.END)

        if(self.mframe[0] != 0 and self.mframe[1] != 0 and self.mframe[2] != 0 and self.mframe[3] != 0):
            self.mframe[0].destroy()
            self.mframe[1].destroy()
            self.mframe[2].destroy()
            self.mframe[3].destroy()
            self.mframe[4].destroy()
            self.mframe[5].destroy()
            self.mframe[6].destroy()
            self.mframe[7].destroy()
            self.mframe[8].destroy()
            self.mframe[9].destroy()

        frame1 = tk.Frame(self.root)
        self.notebook.add(frame1, text="Method_One")
        self.canvas = tk.Canvas(frame1, width=360, height=800, bg="white")
        scroll = tk.Scrollbar(frame1, command=self.canvas.yview)
        self.canvas.config(yscrollcommand=scroll.set, scrollregion=self.canvas.bbox("all"))
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        frameOne = tk.Frame(self.canvas, width=320, height=700, bg="white")
        self.canvas.create_window(C_MENU_POS_X+200, 30, window=frameOne, anchor="n")
        self.canvas.bind_all('<MouseWheel>', self.event_onMouseWheel)

        frame2 = tk.Frame(self.root)
        self.notebook.add(frame2, text="Method_Two")
        self.canvas2 = tk.Canvas(frame2, width=360, height=800, bg="white")
        scroll2 = tk.Scrollbar(frame2, command=self.canvas2.yview)
        self.canvas2.config(yscrollcommand=scroll2.set, scrollregion=self.canvas2.bbox("all"))
        self.canvas2.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scroll2.pack(side=tk.RIGHT, fill=tk.Y)
        frameTwo = tk.Frame(self.canvas2, width=320, height=700, bg="white")
        self.canvas2.create_window(C_MENU_POS_X + 180, 30, window=frameTwo, anchor="n")
        self.canvas2.bind_all('<MouseWheel>', self.event_onMouseWheel)

        frame3 = tk.Frame(self.root)
        self.notebook.add(frame3, text="Method_Three")
        self.canvas3 = tk.Canvas(frame3, width=360, height=800, bg="white")
        scroll3 = tk.Scrollbar(frame3, command=self.canvas3.yview)
        self.canvas3.config(yscrollcommand=scroll3.set, scrollregion=self.canvas3.bbox("all"))
        self.canvas3.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scroll3.pack(side=tk.RIGHT, fill=tk.Y)
        frameThree = tk.Frame(self.canvas3, width=320, height=700, bg="white")
        self.canvas3.create_window(C_MENU_POS_X + 150, 30, window=frameThree, anchor="n")
        self.canvas3.bind_all('<MouseWheel>', self.event_onMouseWheel)

        frame4 = tk.Frame(self.root)
        self.notebook.add(frame4, text="Method_Four")
        self.canvas4 = tk.Canvas(frame4, width=360, height=1400, bg='white')
        scroll4 = tk.Scrollbar(frame4, command=self.canvas4.yview)
        self.canvas4.config(yscrollcommand=scroll4.set, scrollregion=self.canvas4.bbox("all"))
        # self.canvas4.config(yscrollcommand=scroll4.set, scrollregion=(0, 0, 0, 1400))
        self.canvas4.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scroll4.pack(side=tk.RIGHT, fill=tk.Y)
        frameFour = tk.Frame(self.canvas4, width=350, height=1000, bg="white")
        self.canvas4.create_window(C_MENU_POS_X + 180, 30, window=frameFour, anchor="n")
        self.canvas4.bind_all('<MouseWheel>', self.event_onMouseWheel)

        frame5 = tk.Frame(self.root)
        self.notebook.add(frame5, text="Method_Five")
        self.canvas5 = tk.Canvas(frame5, width=360, height=800, bg='white')
        scroll5 = tk.Scrollbar(frame5, command=self.canvas5.yview)
        self.canvas5.config(yscrollcommand=scroll5.set, scrollregion=self.canvas5.bbox("all"))
        self.canvas5.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scroll5.pack(side=tk.RIGHT, fill=tk.Y)
        frameFive = tk.Frame(self.canvas5, width=350, height=700, bg="white")
        self.canvas5.create_window(C_MENU_POS_X+200, 30, window=frameFive, anchor="n")
        self.canvas5.bind_all('<MouseWheel>', self.event_onMouseWheel)

        self.mframe = [frameOne, frameTwo, frameThree, frameFour, frameFive, frame1, frame2, frame3, frame4, frame5]

        self.notebook.select(self.mframeIdx)
        self.root.config(menu=self.menubar)

        self.button_dict = {}
        self.button_skip = {}
        self.filename = ""
        self.tresult = pd.DataFrame()
        self.tresult_base_pos = pd.DataFrame()
        self.alert_msg("")

    def help_about(self):
        tHLayer = tk.Tk()
        tHLayer.title(C_MENU_MAINTITLE)
        # self.root.geometry("640x400+100+100")
        tHLayer.geometry("%dx%d+%d+%d"%(C_MENU_WIDTH/1.8,(C_MENU_HEIGHT-140)/1.8,C_MENU_POS_X*2,C_MENU_POS_Y*2 ))
        tHLayer.resizable(True, True)

        label1 = tk.Label(tHLayer, text="Automatic position recovery\n using relative coordinate on 3D",width=30, height=3)
        label1.grid(row=2, column = 0)

        label2 = tkinter.Label(tHLayer, text="3차원공간 위의 상대좌표를 이용한 자동 위치 복원", width=50, height=5)
        label2.grid(row=3, column = 0)

        label3 = tkinter.Label(tHLayer, text="3차원 공간에 움직이지 않는 기준점(테이블)을 두고,\n사과와 컵의 위치를 각각 추출했을때,\n사과와 컵의 위치 관계를 자동으로 복원시켜주는 기능")
        label3.grid(row=15, column = 0)

        label4 = tkinter.Label(tHLayer, text="Copyright@ 2019-2020 magicst3@gmail.com",width=50, height=5)
        label4.grid(row=18, column = 0, sticky=tk.S)

        tHLayer.mainloop()

    def event_selectTab(self, event):
        # print(event.widget.index("current"))
        self.mframeIdx = event.widget.index("current")
        self.menu_load(autoLoad=1)

    def event_onMouseWheel(self, event):
        """
        A function to handle the onMouseWheel event for different tabs.
        
        Parameters:
            event: the mouse wheel event
        
        Returns:
            None
        """
        tabIdx = self.mframeIdx
        if(tabIdx == C_TAB1):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif(tabIdx == C_TAB2):
            self.canvas2.yview_scroll(int(-1*(event.delta/120)), "units")
        elif (tabIdx == C_TAB3):
            self.canvas3.yview_scroll(int(-1*(event.delta/120)), "units")
        elif (tabIdx == C_TAB4):
            self.canvas4.yview_scroll(int(-1*(event.delta/120)), "units")
        elif (tabIdx == C_TAB5):
            self.canvas5.yview_scroll(int(-1*(event.delta/120)), "units")

    def dynamic_checkBox(self, button_dict, idx, tText=""):
        """
        A function to dynamically create checkboxes based on the given button dictionary.
        
        Parameters:
            button_dict (dict): A dictionary containing the button information.
            idx (int): The index parameter.
            tText (str): Optional text parameter (default is an empty string).
        """
        self.button_dict = button_dict
        self.button_skip = dict([(str(key).split('\t')[0]+" |Skip", value) for key, value in button_dict.items()])
        # self.button_skip = dict([("Skip | "+ key.split('\t\t')[0]+ key.split('\t\t')[1], value) for key, value in button_dict.items()])

        print(self.button_dict, '\n',self.button_skip)
        row = len(self.button_dict) + 2

        tTitleText = tk.Label(self.mframe[idx], text=tText,fg="blue", bg="#FF0")
        tTitleText.grid(row=0, sticky=tk.N)

        for i, key in enumerate(self.button_dict, 2):
            self.button_dict[key] = tk.IntVar()  # set all values of the dict to intvars
            # set the variable of the checkbutton to the value of our dictionary so that our dictionary is updated each time a button is checked or unchecked
            c = tk.Checkbutton(self.mframe[idx], text=key, variable=self.button_dict[key], bg='white')
            c.grid(row=i, sticky=tk.W)
        for i, key in enumerate(self.button_skip, 2):
            self.button_skip[key] = tk.IntVar()
            if (idx == C_TAB1 or idx == C_TAB2 or idx == C_TAB3):
                d = tk.Checkbutton(self.mframe[idx], text=key, variable=self.button_skip[key], bg='pink')
                d.grid(row=i, column =1)

        tinclude = tk.Button(self.mframe[idx], text='Include', command=self.query_include)
        tinclude.grid(row=row, sticky=tk.W)

        texclude = tk.Button(self.mframe[idx], text='Exclude', command=self.query_exclude)
        texclude.grid(row=row, sticky=tk.E, padx=50)

        self.trunning = tk.Button(self.mframe[idx], text='Running_Result', command=self.running_result)
        self.trunning.grid(row=row + 1, sticky=tk.W)


        # self.progress = tkinter.ttk.Progressbar(self.mframe[idx], orient="horizontal", length=400, mode="determinate")
        # # self.progress.pack()
        # # self.progress.grid(row=row + 5, sticky=tk.N)
        # self.progress.grid(row=row + 5, sticky=tk.S, pady=100)

    def query_include(self):
        """
        A function that iterates through button_dict items and prints key if value is True. 
        Then prints the length of button_skip list if not empty and iterates through button_skip items printing key if value is True. 
        """
        for key, value in self.button_dict.items():
            if value.get():
                print(key)
        print("",'\n---skip---', len(self.button_skip))
        if(len(self.button_skip)):
            for key, value in self.button_skip.items():
                if value.get():
                    print(key)
            print("")

    def query_exclude(self):
        """
        A function to query and print keys based on the values of the button_dict and button_skip dictionaries.
        No parameters or return types specified.
        """
        for key, value in self.button_dict.items():
            if not value.get():
                print(key)
        print("",'\n---skip---', len(self.button_skip))
        if(len(self.button_skip)):
            for key, value in self.button_skip.items():
                if not value.get():
                    print(key)
        print("")

    def running_result(self):
        if(self.tOne.is_alive()):
            print("계산중입니다.")
            self.alert_msg("계산중입니다.")
            return

        self.trunning['state'] = "disable"
        self.counter = 0
        self.tOne = threading.Thread(target=self.running_result_thread)
        self.tOne.daemon = True
        self.tOne.start()
        # t1.join()
        self.progress_start()

    def running_result_thread(self):
        """
        Generate a running result thread that prints the current time, deletes text log, calculates auto recovery 3D points, handles different scenarios based on the frame index, and updates the UI state accordingly.
        """
        print_current_time(funcname())
        self.textlog.delete('1.0', tk.END)

        ret, retData, resultData = self.ObjCtrl.calc_auto_recovery_3d_points(self.tdata, self.mframeIdx, self.button_dict, self.button_skip)
        print_current_time(funcname())
        if(ret == False):
            print(retData)
            self.alert_msg(retData)
            self.trunning['state'] = "normal"
            return
        if(self.mframeIdx == C_TAB1 or self.mframeIdx == C_TAB2 or self.mframeIdx == C_TAB3 or self.mframeIdx == C_TAB4):
            print("계산이 완료되었습니다.")
            self.alert_msg("계산이 완료되었습니다.","green2")
            self.tresult = resultData
        elif(self.mframeIdx == C_TAB5):
            print("C_TAB5가 눌렸습니다.")
            # self.alert_msg("C_TAB5가 눌렸습니다.")
            tType = retData
            ret, result = self.ObjCtrl.calc_relative_position_on_base_type(tType, resultData)
            if (ret == True):
                print("계산이 완료되었습니다.")
                self.alert_msg("계산이 완료되었습니다.", "green2")
                self.tresult_base_pos = result

        self.trunning['state'] = "normal"

    def menu_load(self, autoLoad=0):
        """
        A function to load a menu with an option for auto-loading.
        
        Parameters:
        - autoLoad: int, default 0
        
        Returns:
        None
        """
        if(autoLoad == 0):
            self.menu_new() #초기화
            self.filename = filedialog.askopenfilename(initialdir='./', title='Select file',
                                                  filetypes=(("txt files", "*.txt"), ("all files", "*.*")))

        if self.filename:
            try:
                print_current_time(funcname())
                print("Load %s" % self.filename)
                self.textlog.delete('1.0', tk.END)
            except:  # <- naked except is a bad idea
                messagebox.showerror("Load Data File", "Failed to read file\n'%s'" % self.filename)

            tempdata = self.ObjCtrl.load_DPA_file(self.filename)
            self.tdata = tempdata.copy()
            ret_type_one, ret_type_two, ret_type_three, ret_type_four = self.ObjCtrl.preprocess(tempdata)
            # print(ret_type_one)
            # print(ret_type_one.group_sub)
            # print(ret_type_one.title)
            # old_tdic = dict(ret_type_one.group_sub)

            old_tdic_mix = {key + '\t\t|' + value: value for key, value in dict(ret_type_two).items()}
            # print("old_tdic_mix",old_tdic_mix)
            old_tdic_mix2 = dict(ret_type_one.group_sub+'\t\t|'+ret_type_one.title)
            # print('old_tdic_mix3', old_tdic_mix3)
            swap_tdic_mix2 = dict([(value, key) for key, value in old_tdic_mix2.items()])
            old_tdic_mix4 = dict(ret_type_four.title + '\t!' + ret_type_four.point_name+'!'
                     +ret_type_four.tx.astype(str)+','+ret_type_four.ty.astype(str)+','+ret_type_four.tz.astype(str))
            swap_tdic_mix4 = {value:key for key, value in old_tdic_mix4.items()}
            # print("old_tdic_mix4", old_tdic_mix4)

            #키는 중복될수 있어, Value로 데이터를 보내려함
            if(self.mframeIdx == C_TAB1):
                self.dynamic_checkBox(old_tdic_mix, self.mframeIdx, '계산시 기준으로 설정할 포인트 그룹을 선택하세요 |중복개수')
            elif(self.mframeIdx == C_TAB2):
                self.dynamic_checkBox(swap_tdic_mix2, self.mframeIdx, '계산시 기준으로 설정할 포인트 그룹을 선택하세요 |프로젝트명')
            elif(self.mframeIdx == C_TAB3):
                self.dynamic_checkBox(swap_tdic_mix2, self.mframeIdx, '계산시 삭제할 포인트 그룹을 선택하세요 |프로젝트명')
            elif(self.mframeIdx == C_TAB4):
                self.dynamic_checkBox(swap_tdic_mix4, self.mframeIdx, '계산시 삭제할 점들을 선택하세요 |프로젝트명|포인트그룹|좌표')
            elif (self.mframeIdx == C_TAB5):
                self.dynamic_checkBox(old_tdic_mix, self.mframeIdx, '기준과 나머지 그룹의 거리를 추출하려합니다. 기준을 선택하세요 |중복개수')
            return

    def menu_save(self):
        """
        A function to save menu data to a file. 
        """
        if((self.mframeIdx != C_TAB5 and self.tresult.empty is True) or (self.mframeIdx == C_TAB5 and self.tresult_base_pos.empty is True)):
            self.alert_msg('저장할 데이터가 없습니다.')
            return

        filename = filedialog.asksaveasfilename(initialdir='./', title='Select file',
                                                filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
        if filename:
            try:
                print("Save %s" % filename)
            except:  # <- naked except is a bad idea
                messagebox.showerror("Save Data File", "Failed to save file\n'%s'" % filename)

            if (self.mframeIdx == C_TAB5):
                self.ObjCtrl.save_DPA_file(self.tresult_base_pos, filename)
            else:
                self.ObjCtrl.save_DPA_file(self.tresult, filename)
            return

    def menu_save_log(self):
        """
        Save the log to a selected file.

        This function prompts the user to select a file to save the log to. It then attempts to save the log to the selected file. If the save is successful, it returns without any value.

        Parameters:
        - self: the instance of the class
        """
        filename = filedialog.asksaveasfilename(initialdir='./', title='Select file',
                                                filetypes=(("log files", "*.log"), ("all files", "*.*")))
        if filename:
            try:
                print("Save log %s" % filename)
            except:  # <- naked except is a bad idea
                messagebox.showerror("Save Data File", "Failed to save file\n'%s'" % filename)
            with open(filename, 'w') as f:
                f.write(self.textlog.get(1.0, tk.END))
                f.close
            return

    def progress_start(self):
        self.ObjCtrl.progress_pos = 0
        self.progress_pos = 0
        self.progress_end = 100
        self.progress["value"] = self.progress_pos
        self.progress["maximum"] = self.progress_end
        self.progress_checking()
        # self.progress.start(100)

    def progress_checking(self):
        '''simulate reading 500 bytes; update progress bar'''
        # self.progress_pos += 10
        self.progress_pos = self.ObjCtrl.progress_pos
        self.progress_end = self.ObjCtrl.progress_end
        # if(self.progress_end < self.progress_pos):
        #     self.progress_pos = self.progress_end
        self.progress["value"] = self.progress_pos
        self.progress["maximum"] = self.progress_end
        # print("AAAAAAAAAAAAAAAAAAAAAAAA",self.progress_pos)
        if self.progress_pos < self.progress_end:
            # read more bytes after 100 ms
            self.progress.after(50, self.progress_checking)

    def main(self):
        """
        A method that binds the event '<<NotebookTabChanged>>' to the event_selectTab method and starts the main event loop.
        """
        self.notebook.bind('<<NotebookTabChanged>>', self.event_selectTab)
        self.root.mainloop()

    # def main2(self):
    #     while 1:
    #         self.root.update_idletasks()
    #         self.root.update()
    #         time.sleep(0.01)
    #     # self.root.after(1000,self.main2)

class PrintLogger(): # create file like object
    def __init__(self, textbox): # pass reference to text widget
        self.textbox = textbox # keep ref

    def write(self, text):
        self.textbox.insert(tk.END, text) # write text to textbox
            # could also scroll to end of textbox here to make sure always visible

    def flush(self): # needed for file like object
        pass

class Logger():
    def __init__(self, textbox, filename='DebugLog-3d.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.textbox = textbox  # keep ref

    def write(self, message):
        self.terminal.write(message)          # stdout
        self.log.write(message)               # fileout
        self.textbox.insert(tk.END, message)  # write text to textbox
        self.textbox.see(tk.END)

    def flush(self): # needed for file like object
        pass

if __name__ == '__main__':
    print("Start 3d accuracy\n")
    if (0):
        sys.stdout = open('DebugLog-3d.txt', 'w')
    gui = mainMenu_GUI()
    print_current_time("Start 3d accuracy\n")
    gui.main()

    # print(1/0)

    # relative_position_based_on_refer_point()
    # print(1/0)

    # relative_position_based_on_many_points()
    # move_origin_from_refer_point()
    # tdata = load_DPA_file("0128_eye_display_coordinate_ext2.txt") #미완성

    # ret_type_one = preprocess(tdata)
    # print(ret_type_one)
    # tresult = auto_recovery_3d_points_on_each_of_coordinate(tdata)
    # save_DPA_file(tresult, "result.txt")
    # check_face_pos_GT_based_on_MRA2_CAD_displaycenter()
    # test()