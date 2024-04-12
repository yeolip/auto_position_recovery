import numpy as np
import math
import cv2
import sys

C_PRINT_ENABLE = 0

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

def funcname():
    return sys._getframe(1).f_code.co_name + "()"

def callername():
    return sys._getframe(2).f_code.co_name + "()"

def rigid_transform_3D(A, B):
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

def m_findTransformedPoints(P_Ref, P_Input, P_Target):
    print("//////////",funcname(),"//////////")
    n_loop = P_Input.shape[0] / P_Ref.shape[0]
    P_Transformed = [];

    for i in range(int(n_loop)):
        R, T = rigid_transform_3D(P_Ref, P_Input)

        print('R',*R, 'T',*T , sep='\n')
        P_Transformed = (R * P_Target.T + T).T

        if(C_PRINT_ENABLE):
            print('ret', *P_Transformed, sep='\n')
    return P_Transformed, R, T

print("Start 3d accuracy\n")

refer_gt = np.asmatrix(
    [
        [550.0, 0.0   ,  200.0],
        [550.0, -125  ,  200.0],
        [550.0, 125   ,  200.0],
        [660.0, 0.0   ,  200.0],
        [660.0, -157.0,  200.0],
        [660.0, 157.0 ,  200.0],
        [770.0, 0.0   ,  200.0],
        [770.0, -187.0,  200.0],
        [770.0, 187.0 ,  200.0],
        [1000.0, 0.0   , 200.0],
        [1000.0, -223.0, 200.0],
        [1000.0, 223.0 , 200.0]
    ]
)

testBR_gt = np.asmatrix(
    [
        [544.155, 0.301802, 201.1413],
        [542.8000949, -124.592508, 200.8760293],
        [543.6006888, 125.0055653, 201.5867042],
        [653.5253622, -0.065801722, 200.8968923],
        [654.3760825, -156.4375755, 200.1750861],
        [653.8184373, 156.4768642, 201.5808448],
        [764.6075551, -0.332995284, 200.1053119],
        [763.5405458, -186.6702368, 199.8520603],
        [764.1214187, 187.5501312, 201.5968827],
        [994.0503402, -0.484057969, 199.6996862],
        [993.6678989, -223.0837897, 198.6245694],
        [994.8557441, 224.1328336, 199.3728507]
    ]
)

testBR2_gt = np.asmatrix(
    [
        [550.6861, 0.443974, 200.1439],
        [548.9661, -124.446, 199.9525],
        [550.4957, 125.1491, 200.5096],
        [660.0552, -0.2442, 200.2738],
        [660.4501, -156.618, 199.6529],
        [660.8045, 156.2974, 200.8607],
        [771.1381, -0.83724, 199.8621],
        [769.526, -187.171, 199.7219],
        [771.1974, 187.0476, 201.2344],
        [1000.579, -1.66051, 200.241],
        [999.5485, -224.259, 199.3039],
        [1002.044, 222.9527, 199.7762]
    ]
)

mra2_gt = np.asmatrix(
    [
        [549.9052842, 0.091194222, 200.2352574],
        [550.277942, -124.7490661, 200.4497027],
        [549.5165888, 124.1982775, 200.0850256],
        [658.9904336, 0.539462538, 200.2907827],
        [659.5925395, -155.8197885, 200.7203268],
        [658.4953224, 157.1691639, 200.0802675],
        [770.3391025, 0.391250747, 200.3953258],
        [770.8565335, -186.492064, 200.675229],
        [769.7610079, 187.444418, 200.0141111],
        [1000.662198, 0.383123549, 200.152945],
        [1001.469656, -222.1727999, 200.8050351],
        [1000.325796, 223.4947794, 199.9249182]
    ]
)

br214_gt = np.asmatrix(
    [
        [549.8702793, -0.778261092, 201.3407507],
        [550.1499817, -125.6168456, 200.6527736],
        [549.5740919, 124.3267282, 202.0876683],
        [658.9556302, -0.410765481, 201.2283087],
        [659.4415633, -156.7691455, 200.5273751],
        [658.5772363, 156.2164433, 202.1500513],
        [770.3041269, -0.641744778, 201.1570219],
        [770.6823414, -187.5222328, 200.0860819],
        [769.8652175, 186.4093459, 202.1279754],
        [1000.626372, -0.817868779, 200.5531075],
        [1001.268538, -223.3728012, 199.5961945],
        [1000.45634 ,222.289461, 201.937348]
    ]
)
br214_gt_02 = np.asmatrix(
    [
        [550.437 , -1.77578, 201.0105],
        [550.4987, -126.014, 200.2275],
        [550.3576,  122.9289,    201.8519],
        [659.5227, -0.9979,     200.7912],
        [659.7357, -157.356,    199.9711],
        [659.4179, 155.6289, 201.8322],
        [770.8706, -1.42247 , 200.6104],
        [770.9224, -188.302, 199.3972],
        [770.7583, 185.6282, 201.7237],
        [1001.192, -1.99862, 199.780],
        [1001.445, -224.553, 198.6536],
        [1001.411 ,221.1074, 201.3339]
    ]
)

eva23d_test = np.asmatrix(
    [
        [552.0080053, -7.265726888, 184.7465286],
        [550.8455845, -132.1004894, 184.3317757],
        [553.1455731, 116.8358053, 185.2218572],
        [661.0900501, -8.157872908, 184.4543353],
        [659.770005, -164.512692, 184.0953879],
        [662.5210982, 148.4649815, 185.0333254],
        [772.4282486, -9.674493901, 184.2008967],
        [770.6475719, -196.5488797, 183.5390335],
        [774.1500369, 177.3709725, 184.7624991],
        [1002.731913, -12.51092814, 183.2195445],
        [1000.803585, -235.0601345, 182.7494834],
        [1005.139431, 210.58609, 184.1149464]
    ]
)




# primeP2, rr, tt = m_findTransformedPoints(eva23d_test, mra2_gt, eva23d_test)
# primeP2, rr, tt = m_findTransformedPoints(br214_gt_02, mra2_gt, br214_gt_02)
primeP2, rr, tt = m_findTransformedPoints(testBR2_gt, refer_gt, testBR2_gt)

print('\nTranslation',tt)
print('Rodrigues - R31(deg)', cv2.Rodrigues(rr)[0] * radianToDegree)
print('Euler - R31(deg)', rotationMatrixToEulerAngles(rr) * radianToDegree)
print('\nafter apply R,T, then\n',primeP2)


