import numpy as np
import math
import cv2
import csv
import pandas as pd

#Automatic virtual position recovery using relative coordinates

# 최대 줄 수 설정
pd.set_option('display.max_rows', 1000)
# 최대 열 수 설정
pd.set_option('display.max_columns', 50)
# 표시할 가로의 길이
pd.set_option('display.width', 120)

degreeToRadian = math.pi/180
radianToDegree = 180/math.pi

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

# a= np.array((0.00	,0.00	,0.00))
# b= np.array((3.00	,0.00	,0.00))
# c= np.array((3.00   ,3.00   ,0.00))
# d= np.array((0.00   ,3.00   ,0.00))
#
# e= np.array(( 9.00   ,3.00   ,0.00))
# f= np.array((18.00   ,6.00   ,0.00))
# g= np.array(( 6.00   ,21.00   ,0.00))
# print(math.acos(1/1.414) * 180/math.pi)
# h= b-a
# i= c-a
# j= d-a
# print('h',h)
# print('i',i)
# print('j',j)
# print(np.dot(h, i))
# print(np.cross(h, i))
# print(np.dot(h, j))
# print(np.cross(h, j))
# test_cross = np.cross(h, j)
# # 0 = dot product with cross vector using every vector in char
# print('%.10f'% np.dot(test_cross, a-e))
# print('%.10f'% np.dot(test_cross, b-a))
# print('%.10f'% np.dot(test_cross, c-a))
# print('%.10f'% np.dot(test_cross, d-a))
# print('%.10f'% np.dot(test_cross, e-a))
# print('%.10f'% np.dot(test_cross, f-a))
# print('%.10f'% np.dot(test_cross, g-a))
# print('%.10f'% np.dot(test_cross, f-e))
# print('%.10f'% np.dot(test_cross, g-c))
#
# print('각도',math.acos(np.dot(test_cross, a)) * 180 / math.pi)
# print('jawha chart','*'*50)
#
# a= np.array((0.079311	,0.174912	,0.755718))
# b= np.array((0.049689	,0.175313	,0.750517))
# c= np.array((0.049247	,0.145295	,0.750445))
# d= np.array((0.07886 ,0.144896, 0.75574))
#
# e= np.array((0.04876		,0.115213	,0.750144))
# f= np.array((0.048259	,0.085176	,0.749968))
# g= np.array((-0.159897	,-0.001663	,0.714354))
# print(math.acos(1/1.414) * 180/math.pi)
# h= b-a
# i= c-a
# j= d-a
# print('h',h)
# print('i',i)
# print('j',j)
# print(np.dot(h, i))
# print(np.cross(h, i))
# print(np.dot(h, j))
# print(np.cross(h, j))
# test_cross = np.cross(h, j)
# # 0 = dot product with cross vector using every vector in char
# print('%.10f'% np.dot(test_cross, a-e))
# print('%.10f'% np.dot(test_cross, b-a))
# print('%.10f'% np.dot(test_cross, c-a))
# print('%.10f'% np.dot(test_cross, d-a))
# print('%.10f'% np.dot(test_cross, e-a))
# print('%.10f'% np.dot(test_cross, f-a))
# print('%.10f'% np.dot(test_cross, g-a))
# print('%.10f'% np.dot(test_cross, f-e))
# print('%.10f'% np.dot(test_cross, g-c))
# print('각도',math.acos(np.dot(test_cross, a)) * 180 / math.pi)
# print('*'*50)
#
#
# print('sm chart - objectpoint3D_1.txt','*'*50)
#
# a= np.array((0.070856,  -0.141501,  0.746554))
# b= np.array((0.082736,  -0.139778,  0.746300))
# c= np.array((0.081050,  -0.129107,  0.751628))
# d= np.array((0.069249,  -0.130671,  0.751461))
#
# e= np.array((0.134306,  -0.001547,  0.811589))
# f= np.array((0.176706,  -0.039356,  0.791868))
# g= np.array((0.115216,  -0.113509,  0.758401))
# h= b-a
# i= c-a
# j= d-a
# print('h',h)
# print('i',i)
# print('j',j)
# print(np.dot(h, i))
# print(np.cross(h, i))
# print(np.dot(h, j))
# print(np.cross(h, j))
# test_cross = np.cross(h, j)
# # 0 = dot product with cross vector using every vector in char
# print('%.10f'% np.dot(test_cross, a-e))
# print('%.10f'% np.dot(test_cross, b-a))
# print('%.10f'% np.dot(test_cross, c-a))
# print('%.10f'% np.dot(test_cross, d-a))
# print('%.10f'% np.dot(test_cross, e-a))
# print('%.10f'% np.dot(test_cross, f-a))
# print('%.10f'% np.dot(test_cross, g-a))
# print('%.10f'% np.dot(test_cross, f-e))
# print('%.10f'% np.dot(test_cross, g-c))
# print('각도',math.acos(np.dot(test_cross, a)) * 180 / math.pi)
# print('*'*50)
#
# print('9x9 circle grid chart - objectpoint3D_2.txt','*'*50)
#
# a= np.array((-0.059502,  -0.155606,  0.853068))
# b= np.array((-0.029504,  -0.155477,  0.853454))
# c= np.array((-0.029614,  -0.125439,  0.853789))
# d= np.array((-0.059638,  -0.125603,  0.853702))
#
# e= np.array((0.119562,  0.054919,  0.859973))
# f= np.array((0.059799,  -0.005236,  0.857721))
# g= np.array((0.154781,  -0.156280,  0.859369))
# h= b-a
# i= c-a
# j= d-a
# print('h',h)
# print('i',i)
# print('j',j)
# print(np.dot(h, i))
# print(np.cross(h, i))
# print(np.dot(h, j))
# print(np.cross(h, j))
# test_cross = np.cross(h, j)
# # 0 = dot product with cross vector using every vector in char
# print('%.10f'% np.dot(test_cross, a-e))
# print('%.10f'% np.dot(test_cross, b-a))
# print('%.10f'% np.dot(test_cross, c-a))
# print('%.10f'% np.dot(test_cross, d-a))
# print('%.10f'% np.dot(test_cross, e-a))
# print('%.10f'% np.dot(test_cross, f-a))
# print('%.10f'% np.dot(test_cross, g-a))
# print('%.10f'% np.dot(test_cross, f-e))
# print('%.10f'% np.dot(test_cross, g-c))
# print('각도',math.acos(np.dot(test_cross, a)) * 180 / math.pi)
# print('*'*50)
#
# print('projection plane - 9x9 circle grid chart - objectpoint3D_2.txt','*'*50)
#
# a= np.array((0.0607760881,  -0.0301225983,  0.8439268045))
# b= np.array((0.0906037743, -0.0300211148,  0.8443397305))
# c= np.array((0.0904271574, -0.0002437271,  0.8448646253))
# d= np.array((0.0606141817, -0.0003490928,  0.8444518333))
#
# e= np.array((0.1496979683,  0.0594596900,  0.8467388733))
# f= np.array((0.0305134170,  0.0590399745,  0.8450886644))
# g= np.array((0.0899975801,  0.0890212626,  0.8464395024))
# h= b-a
# i= c-a
# j= d-a
# print('h',h)
# print('i',i)
# print('j',j)
# print(np.dot(h, i))
# print(np.cross(h, i))
# print(np.dot(h, j))
# print(np.cross(h, j))
# test_cross = np.cross(h, j)
# # 0 = dot product with cross vector using every vector in char
# print('%.10f'% np.dot(test_cross, a-e))
# print('%.10f'% np.dot(test_cross, b-a))
# print('%.10f'% np.dot(test_cross, c-a))
# print('%.10f'% np.dot(test_cross, d-a))
# print('%.10f'% np.dot(test_cross, e-a))
# print('%.10f'% np.dot(test_cross, f-a))
# print('%.10f'% np.dot(test_cross, g-a))
# print('%.10f'% np.dot(test_cross, f-e))
# print('%.10f'% np.dot(test_cross, g-c))
# print('각도',math.acos(np.dot(test_cross, a)) * 180 / math.pi)
# print('*'*50)
#
#
# print('projection plane - sm chart - objectpoint3D_1.txt','*'*50)
#
# a= np.array((0.0708468880, -0.1413858742,  0.7463172135))
# b= np.array((0.0827462198, -0.1399074094,  0.7465671366))
# c= np.array((0.0810573100, -0.1292012879,  0.7518220414))
# d= np.array((0.0692524020, -0.1307146590,  0.7515514985))
#
# e= np.array((0.1152056104, -0.1133864644,  0.7581475720))
# f= np.array((0.2085738696, -0.0893623254,  0.7661295196))
# g= np.array((0.0695950514, -0.0543322026,  0.7885565041))
# h= b-a
# i= c-a
# j= d-a
# print('h',h)
# print('i',i)
# print('j',j)
# print(np.dot(h, i))
# print(np.cross(h, i))
# print(np.dot(h, j))
# print(np.cross(h, j))
# test_cross = np.cross(h, j)
# # 0 = dot product with cross vector using every vector in char
# print('%.10f'% np.dot(test_cross, a-e))
# print('%.10f'% np.dot(test_cross, b-a))
# print('%.10f'% np.dot(test_cross, c-a))
# print('%.10f'% np.dot(test_cross, d-a))
# print('%.10f'% np.dot(test_cross, e-a))
# print('%.10f'% np.dot(test_cross, f-a))
# print('%.10f'% np.dot(test_cross, g-a))
# print('%.10f'% np.dot(test_cross, f-e))
# print('%.10f'% np.dot(test_cross, g-c))
# print('각도',math.acos(np.dot(test_cross, a)) * 180 / math.pi)
# print('*'*50)

def m_ProjectUserCoor(input_point, ref):
    print("m_ProjectUserCoor")

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
    print('ret',*output_point, sep='\n')
    return output_point, t_mean, t_axis


def m_ProjectAbsCoor(P_DPA_Pts, P_DPA_Ref1):
    print("m_ProjectAbsCoor")
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

    Centroid, Axis = m_MakeAbsAxis(P_DPA_Ref1);
    # print(P_DPA_Pts, np.tile(Centroid, (P_DPA_Pts.shape[0],1)),'\n',Axis )
    P_Ref1_Pts = (P_DPA_Pts - np.tile(Centroid, (P_DPA_Pts.shape[0],1))) * Axis

    print('ret',*P_Ref1_Pts, sep='\n')
    # P_Ref1_Pts =
    # 1347.73296211896    404.477604521447 - 5.85905452814663
    # 1347.61341605657    311.995327466233    212.156432418129
    # 1347.38468857793    442.937470819739    266.846311762761
    # 1407.27742564632    506.339726820845    163.150085194833
    # 1347.48922525842    532.220488679192    47.3366148858555
    # np.tile(T, (1, P_Ref.shape[0]))
    return P_Ref1_Pts


def m_ProjectDispCoor(P_Ref1_Pts, P_Ref1_Disp):
    print("m_ProjectDispCoor")
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
    Centroid, Axis = m_MakeDispAxis(P_Ref1_Disp);

    # Centroid = np.asmatrix([260.6765,  418.9962,  126.6164])
    # Axis = np.asmatrix(
    # [[0.999957259960338, 2.426512344112703e-04, 0.009242267266263],
    # [0.008617504541717, -0.386586756527017, -0.922224100117558],
    # [0.003349159310558, 0.922253013439533, -0.386545326783471]]
    # )
    P_Disp = (P_Ref1_Pts - np.tile(Centroid, (P_Ref1_Pts.shape[0],1))) * Axis

    print('ret',P_Disp)
    return P_Disp, Centroid, Axis


def rigid_transform_3D(A, B):
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

def m_findTransformedPoints(P_Ref, P_Input, P_Target):
    print("m_findTransformedPoints")
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

        R, T = rigid_transform_3D(P_Ref, P_Input)

        print('R',*R, 'T',*T , sep='\n')
        # print(P_Target, P_Target.shape)
        # P_Transformed = [P_Transformed;(R * P_Target'+T)'];
        P_Transformed = (R * P_Target.T + T).T
        # P_Transformed = (R * P_Target.T) + np.tile(T, (1, P_Ref.shape[0]))
        # P_Transformed = P_Transformed.T
        # P_Transformed = (R * P_Target[0].T + T.T )
        # P_Transformed[1] = (R * P_Target[1].T + T.T )
        # P_Transformed[2] = (R * P_Target[2].T + T.T )

        print('ret', *P_Transformed, sep='\n')
        # print('ret', P_Transformed)
        # P_Transformed =
        # 535.439643764020, - 51.8884288910241,  63.5738272867725
        # 1012.14066152380, - 47.4931985302768, - 481.028802658092
        # 792.075892677580, - 46.8164625429134, - 671.160246228224
        # 293.693512131688, - 51.1847826907368, - 145.560518603204
    return P_Transformed






# OK
def m_MakeAbsAxis(P_DPA_Ref1):
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

def m_MakeDispAxis(P_Ref1_Disp):
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


# a = np.random.randn(9, 6) + 1j * np.random.randn(9, 6)
# b = np.random.randn(2, 7, 8, 3) + 1j * np.random.randn(2, 7, 8, 3)
# u, s, vh = np.linalg.svd(a, full_matrices=True)
#
# print('\na', a,a.shape)
# print(u.shape, s.shape, vh.shape)
# print('\nu', u, '\ns', s, '\nvh', vh)
#
# print(np.allclose(a, np.dot(u[:, :6] * s, vh)))
#
# smat = np.zeros((9, 6), dtype=complex)
# smat[:6, :6] = np.diag(s)
# print(np.allclose(a, np.dot(u, np.dot(smat, vh))))

def relative_position_based_on_refer_point():
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
    p550N_Eye = m_findTransformedPoints(pMane_Eye_disp[16:19], p550N_name[9:12], pMane_Eye_disp[13:15] )
    print('p550N_Eye\n', p550N_Eye)
                                            #DISP                #DISP            #PATTERN
    p550N_Pattern = m_findTransformedPoints(pPattern_disp[0:5] , p550N_name[4:9], pPattern_disp[5:10] )
    print('p550N_Pattern\n', p550N_Pattern)

    m_ProjectDispCoor(p550N_Eye, p550N_Pattern)
    m_ProjectDispCoor(np.mean(p550N_Eye, axis=0), p550N_Pattern)
    return

def relative_position_based_on_many_points():
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
    primeP2 = m_findTransformedPoints(p660P,p550N,p660P_Full)
    primeP = m_findTransformedPoints(p550N,p660P,p550N_Full)

    print('\n\n')
    print('p550N_Full - primeP2', *(p550N_Full - primeP2), sep='\n')
    print('p660P_Full - primeP', *(p660P_Full - primeP), sep='\n')
    print('\n\n')
    #
    # primeP3 = m_findTransformedPoints(p660P, p660P_base501, p660P)
    #
    # primeP4 = m_findTransformedPoints(p660P, p660P_base503, p660P_Full)

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
    ret1 = m_ProjectAbsCoor(p660P_Full, p660P)
    # m_ProjectAbsCoor(p660P_Full-np.asmatrix(np.ones((p660P_Full.shape[0],1))*[170.573,-1.146,	-1.093]), p660P_base503)
    ret2 = m_ProjectAbsCoor(p550N_Full, p550N)

    print('ret1', ret1)
    print('ret2', ret2)
    print('Sub1', *(ret1 - ret2), sep='\n')

    ret4 = m_ProjectAbsCoor(p660P_Full, p660P_Full[18:22])
    ret5 = m_ProjectAbsCoor(p550N_Full, p550N_Full[18:22])

    print('ret4', *ret4, sep='\n' )
    print('ret5', *ret5, sep='\n')
    print('Sub2', *(ret4 - ret5), sep='\n')

    #recovery original position
    ret6 = m_findTransformedPoints(ret4[18:22], p660P_Full[18:22], ret4)
    ret7 = m_findTransformedPoints(ret5[18:22], p550N_Full[18:22], ret5)

    print('ret6', *ret6, sep='\n' )
    print('ret7', *ret7, sep='\n')
    print('Sub3', *(ret6 - ret7), sep='\n')

def move_origin_from_refer_point():
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

    # ret = m_findTransformedPoints(pTest[0:5], p660P_to_origin, pTest)
    # print('final_ret', *ret, sep='\n')
    retData, t_mean2, test_axis2 = m_ProjectUserCoor(pTest, pTest[5:10])

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

def load_DPA_file(fname):
    tData = []
    # fp = open(fname, 'r', encoding='utf-8')
    # for row in csv.reader(fp, skipinitialspace= True, delimiter=','):
    #     print(row)
    #     # print(row[0], row[1], row[2], row[3], row[4], row[5])
    #     tData.append([row[0], row[1], row[2], row[3], row[4], row[5]])
    # fp.close()
    #
    # tData.sort(key=lambda x: (x[0], x[2], x[1]))
    # # sorted(tData, key=lambda x: (x[0], -x[2], x[1]))
    # for row in tData:
    #     # print(row)
    #     print(row[0], row[1], row[2], row[3], row[4], row[5])
    # print(tData[0:-1])

    df = pd.read_csv(fname, skipinitialspace = True, header=None)
    # print(df.head())
    df.columns = ['title', 'number', 'point_name', 'tx', 'ty', 'tz']
    df = df.sort_values(['title','point_name','number'], ascending=(True,True,True))
    print(tData)

    return df

def save_DPA_file():
    pass

def auto_recovery_3d_points_on_each_of_coordinate_predebug(tData):
    # print( str(df['point_name']).split("_") )
    tData['group_first'] = tData.groupby('title').grouper.group_info[0] + 1
    tData['group_sub'] = tData['point_name'].str.split('_').str[0]
    # print(df.groupby('title').count())
    # print(df.groupby('title').grouper.group_info[0]+1)
    # print(df.groupby('group').grouper.group_info[0]+1)
    # print(d)
    # print(df['group'].)
    # df['group'] = \
    # print(str(df['point_name']).split("_"))
    # print(df.index)
    # print(df.groupby('group').grouper.group_info[0]+1)
    # print(df.head())
    print(tData)

    # print( df.groupby(df.title).keys)

    # print( df.groupby('title').size().get_values())
    # print( df.groupby('title').get_group('403589_DISP'))
    # print( df.groupby('title').all())
    # print( df.groupby(df.title).grouper.group_info)
    # print( df.groupby(df.group_sub).keys)
    # print( df.groupby(df.group_sub).grouper.group_info)

    # pieces = dict(list(df.groupby('title')))
    # print('pieces',pieces)
    print("**" * 50)
    # print(df[['tx','ty','tz']])

    #    # Title/Labeling 구별
    for tkey, tdata in tData.groupby(['title','group_sub']):
       # print(type(tkey),'\n',type(tdata))
       # print(tkey['group_sub'])
       if(tkey[1] != '*'):
           print(tkey,'\n',np.array(tdata[['tx','ty','tz']].reset_index(drop=True)))
           # test = np.array(tdata[['tx','ty','tz']])
           # print('test',test)

    # print('_________________________', data[3], data[4], data[5])
    print("&&" * 50)

    # print(set(df.groupby('group_sub').grouper.group_info[0]))
    # print(df.groupby('group_sub', axis=0).size())

    # Labeling이 동일한 Title 추출 (기준점을 찾기위함)
    df3 = tData[['group_sub', 'title']].drop_duplicates()
    df5_list = df3[~df3['group_sub'].str.contains("\*")].reset_index(drop=True)
    print(df5_list)
    df7_title = df5_list.groupby('title').count() \
        .sort_values(['group_sub'], ascending=False)
    print(df7_title)
    df9_group_sub = df5_list.groupby('group_sub').count() \
        .sort_values(['title'], ascending=False)

    # print(df5_list)
    # print(list(df5_list.index))
    # print(list(df5_list.group_sub))
    # print(list(df5_list.group_sub.value_counts()))
    print(list([df5_list.group_sub, df5_list.group_sub.value_counts()]))
    print(df5_list.group_sub.value_counts()[0])
    print(df5_list.group_sub.value_counts().index[0])

    print(df5_list.group_sub.value_counts().index)

    print("\ntest2\n")
    print((df5_list.group_sub))
    tData_grp= []
    for tnum in range(0,len(df5_list.group_sub.value_counts().index),1):
        # print(tnum)
        ta = int(df5_list.group_sub.value_counts()[tnum])
        if(ta > 1 ):
            tb = df5_list.group_sub.value_counts().index[tnum]
            # print(df5_list.group_sub.value_counts().index[tnum], int(df5_list.group_sub.value_counts()[tnum]))
            print(ta, tb)
            tData_grp.append([ta, tb])
    print('tData_grp', tData_grp)
    # df5_list['gcount'] = df5_list.group_sub.value_counts() > 1
    # print(df5_list )

    # print('check',np.array(df5_list))
    # for tidx, tcheck in zip(df5_list.group_sub):
    #     if(tcheck.value_counts()[tidx] > 0):
    #         print(tcheck.value_counts().index[0])
    # for tcheck in df5_list.group_sub.value_counts():
    #     if(tcheck > 0):
    #         print(tcheck.index[0])

    # print(tData[tData.group_sub == 'DISP'].value_counts())

    # df3.drop_duplicates()
    # print('df3',df3)
    # for tkey, tdata in tData.groupby(['group_sub','title']):
    #     if(tkey[0] != '*'):
    #         label_title.append(tkey)
    #     # print(type(tkey),'\n',type(tdata))
    #     # print(tkey['group_sub'])
    #     print(tkey, tdata.count())
    # print(label_title)
    # print("&&" * 50)

    # print( df.title.groupby(df.group_sub).grouper.group_info)
    # print(df['title'].groupby(df.group_sub))

    # print(df)

    #    # Title/구별
    print("//////////////Title 구별//////////////////")
    label_title = []
    for tkey, tdata in tData.groupby(['title']):
       print(tkey)
       # print(type(tkey),'\n',type(tdata))
       # print(tkey['group_sub'])
       # print(tkey,'\n',tdata[['tx','ty','tz']].reset_index(drop=True))
       # print(tdata[['tx','ty','tz']])
       # tdata_list = tdata[~tdata['group_sub'].str.contains("\*")]
       # tdata_list = tdata[df5_list['group_sub'].keys]
       # print(tdata_list)

       # label_title.append(tdata)
       # print(tdata)

    # print('_________________________', data[3], data[4], data[5])
    print("&&" * 50)

    for tidx, trow in tData.iterrows():
        print(tidx)
        print(trow.point_name)

    print([(x,y) for x,y in zip(tData['title'],tData['group_sub']) ])

    print(df9_group_sub['title'])
    print('ttt',df9_group_sub['title'].index)
    df11 = df9_group_sub['title']
    for tidx  in df9_group_sub['title']:
        if(tidx > 1):
            print(tidx)
            print(tidx.index)
            # print(df9_group_sub['group_sub'])
            # print(df9_group_sub['title'].index)
            # df9_group_sub['title']

            # df9_group_sub['title'][tidx].drop(tidx)
            # df11_duplicate = df11_duplicate.append(drop([tidx]))
    # print(df9_group_sub['title'])
    print(df11)

    for x in df9_group_sub.index:
        # print(x)
        # print(tData[tData.group_sub == x])
        for y in tData.itertuples(index=True, name='Pandas'):
            # print(getattr(y, "point_name"), getattr(y, "group_sub"))
            # ta = getattr(y, "point_name")
            # tb = getattr(y, "group_sub")
            if (x == getattr(y, "group_sub")):
                print('match', y)

    # print(label_title)
    # tData.merge()
    # tData.append(a)
    pass
def auto_recovery_3d_points_on_each_of_coordinate(tData):
    tData['group_first'] = tData.groupby('title').grouper.group_info[0] + 1
    tData['group_sub'] = tData['point_name'].str.split('_').str[0]
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


    print("\n[group_sub] 기준으로 (중복 제거) title의 분류된 label의 갯수가 2개이상인 데이터 추출\n")
    # print('total '(df5_list.group_sub))
    tData_grp= []
    for tnum in range(0,len(df5_list.group_sub.value_counts().index),1):
        # print(tnum)
        ta = int(df5_list.group_sub.value_counts()[tnum])
        if(ta > 1 ):
            tb = df5_list.group_sub.value_counts().index[tnum]
            # print(ta, tb)
            tData_grp.append([ta, tb])
    print('tData_grp', tData_grp)

    # print("&&" * 50)
    # #   # Title/Labeling 구별
    # list_type_point = []
    # dict_type_point = {}
    # for tcount, tname in tData_grp:
    #     temp = []
    #     for tkey, tdata in tData.groupby(['title','group_sub']):
    #        # print(type(tkey),'\n',type(tdata))
    #        if(tkey[1] != '*' and tkey[1] == tname):
    #            print(tkey,'\n',np.array(tdata[['tx','ty','tz']].reset_index(drop=True)))
    #            ta = np.array(tdata[['tx','ty','tz']].reset_index(drop=True))
    #            temp = list([tkey[0], ta])
    #            # test = np.array(tdata[['tx','ty','tz']])
    #            # print('test',test)
    #            dict_type_point = dict(tkey[1], temp)
    #            # list_type_point.append([tkey[1], temp])
    #            # print(dict(tkey[1], temp))
    #
    #
    # # print(list_type_point)
    # print(dict_type_point)
    # print("\\\\" * 50)
    #
    # # print(tData[~tData['point_name'].str.contains('\*')].reset_index(drop=True))
    # # df5_list = df3[~df3['group_sub'].str.contains("\*")].reset_index(drop=True)
    # # print(df7_title.index)


    tvalidData = tData[~tData['point_name'].str.contains('\*')].reset_index(drop=True)
    print('tvalidData', tvalidData)
    tmodifiedData = tvalidData


    for tidx in df9_group_sub.index:
        print(tidx)
        for tkey, tdata in tvalidData.groupby(['title']):
            print(tkey,tdata)
            for tkey2, tdata2 in tvalidData.groupby(['title', 'group_sub']):
                print(tkey2, tdata2)
        # for test in df5_list:
        #     if(tidx == tkey):
        #         print(tkey, tdata)
        # print(tidx, tdata[0], tdata[1])
        # for (tcol_title,tcol_group_sub), tdata in tvalidData.groupby(['title', 'group_sub']):
        #     print(tcol_title, tcol_group_sub)
        #     if(tcol_group_sub == tlabel):
        #     if(tkey[1] == )
        #         print(tlabel, '\n', np.array(tdata[['tx', 'ty', 'tz']].reset_index(drop=True)))
    print("TTTTTTTTTTTTTT")


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

    rrrrr = m_findTransformedPoints(aaaa,bbbb,aaaa)
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



    #
    # # print("///" * 50)
    # #
    # # for tcount, tname in tData_grp:
    # #     # print(x)
    # #     # print(tData[tData.group_sub == x])
    # #     for y in tData.itertuples(index=False, name='Pandas'):
    # #         # print(getattr(y, "point_name"), getattr(y, "group_sub"))
    # #         # ta = getattr(y, "point_name")
    # #         # tb = getattr(y, "group_sub")
    # #         if (tname == getattr(y, "group_sub")):
    # #             print('match', y)
    return

def check_face_pos_GT_based_on_MRA2_CAD_displaycenter(): #미완성
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
    p550N_Eye = m_findTransformedPoints(pMane_Eye_disp[16:19], p550N_name[9:12], pMane_Eye_disp[13:15] )
    print('p550N_Eye\n', p550N_Eye)
                                            #DISP                #DISP            #PATTERN
    p550N_Pattern = m_findTransformedPoints(pPattern_disp[0:5] , p550N_name[4:9], pPattern_disp[5:10] )
    print('p550N_Pattern\n', p550N_Pattern)

    p550N_merge = np.concatenate((p550N_name, p550N_Pattern, p550N_Eye, np.mean(p550N_Eye, axis=0) ))

    pDCenter, Rc, Tc = m_ProjectDispCoor(p550N_merge, p550N_Pattern)
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

def test():
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
    # rrrrr = m_findTransformedPoints(mst_739106_03_NG[0:5],mst_739106_02_NG[0:5],mst_739106_03_NG)
    tR, tT = rigid_transform_3D(mst_739106_03_NG, mst_739106_02_NG)
    print('tR33', *tR, 'tT(mm)', *tT, sep='\n')
    print('R31(deg)', cv2.Rodrigues(tR)[0] * radianToDegree)
    # print('rrrrr\n', rrrrr)

    tR2, tT2 = rigid_transform_3D(mst_739106_03_NG, mst_739106_01_NG)
    print('tR33', *tR2, 'tT(mm)', *tT2, sep='\n')
    print('R31(deg)', cv2.Rodrigues(tR2)[0] * radianToDegree)

    print("mst_739066_02_NG 기준")
    tR, tT = rigid_transform_3D(mst_739066_02_NG, mst_739066_01_NG)
    print('tR33', *tR, 'tT(mm)', *tT, sep='\n')
    print('R31(deg)', cv2.Rodrigues(tR)[0] * radianToDegree)

    print("mst_739095_02_OK 기준")
    tR, tT = rigid_transform_3D(mst_739095_02_OK, mst_739095_01_OK)
    print('tR33', *tR, 'tT(mm)', *tT, sep='\n')
    print('R31(deg)', cv2.Rodrigues(tR)[0] * radianToDegree)

    print("mst_739096_02_OK 기준")
    tR, tT = rigid_transform_3D(mst_739096_02_OK, mst_739096_01_OK)
    print('tR33', *tR, 'tT(mm)', *tT, sep='\n')
    print('R31(deg)', cv2.Rodrigues(tR)[0] * radianToDegree)


# print("Sub\n", *(m_ProjectAbsCoor(p660P_Full, p660P)-m_ProjectAbsCoor(p660P_Full-np.asmatrix(np.ones((p660P_Full.shape[0],1))*[170.573,-1.146,	-1.093]), p660P_base503)), sep='\n' )
# print("Sub\n", *(m_ProjectAbsCoor(p660P_Full, p660P)-m_ProjectAbsCoor(p550N_Full, p550N)), sep='\n' )
# m_ProjectAbsCoor(p660P_Full, p660P_base501)

# print('next\n')
# m_ProjectAbsCoor(m_ProjectAbsCoor(P_DPA_Pts, P_DPA_Ref1), -P_DPA_Ref1)
#
#
# m_findTransformedPoints(P_DPA_Pts,m_ProjectAbsCoor(P_DPA_Pts, P_DPA_Ref1),np.zeros((5,3)))
print("Start 3d accuracy\n")
# relative_position_based_on_refer_point()
# relative_position_based_on_many_points()
# move_origin_from_refer_point()
tdata = load_DPA_file("0128_eye_display_coordinate.txt") #미완성
auto_recovery_3d_points_on_each_of_coordinate(tdata)
# check_face_pos_GT_based_on_MRA2_CAD_displaycenter()
# test()