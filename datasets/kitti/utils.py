import numpy as np


def velo2cam():
    R = np.array([
        7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
        -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
    ]).reshape(3, 3)
    T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    velo2cam = np.hstack([R, T])
    velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
    return velo2cam


def get_relative_pose(pose_1, pose_2):
    # as seen in https://github.com/chrischoy/FCGF
    M = (velo2cam() @ pose_1.T @ np.linalg.inv(pose_2.T) @ np.linalg.inv(velo2cam())).T
    return M
