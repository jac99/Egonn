import numpy as np
import os
from scipy.spatial import distance_matrix

# Faulty point clouds (with 0 points)
FAULTY_POINTCLOUDS = [1566279795718079314]

# Coordinates of test region centres (in Sejong sequence)
TEST_REGION_CENTRES = np.array([[345090.0743, 4037591.323], [345090.483, 4044700.04],
                                [350552.0308, 4041000.71], [349252.0308, 4044800.71]])

# Radius of the test region
TEST_REGION_RADIUS = 500

# Boundary between training and test region - to ensure there's no overlap between training and test clouds
TEST_TRAIN_BOUNDARY = 50


def in_train_split(pos):
    # returns true if pos is in train split
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    dist = distance_matrix(pos, TEST_REGION_CENTRES)
    mask = (dist > TEST_REGION_RADIUS + TEST_TRAIN_BOUNDARY).all(axis=1)
    return mask


def in_test_split(pos):
    # returns true if position is in evaluation split
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    dist = distance_matrix(pos, TEST_REGION_CENTRES)
    mask = (dist < TEST_REGION_RADIUS).any(axis=1)
    return mask


def find_nearest_ndx(ts, timestamps):
    ndx = np.searchsorted(timestamps, ts)
    if ndx == 0:
        return ndx
    elif ndx == len(timestamps):
        return ndx - 1
    else:
        assert timestamps[ndx-1] <= ts <= timestamps[ndx]
        if ts - timestamps[ndx-1] < timestamps[ndx] - ts:
            return ndx - 1
        else:
            return ndx


def read_lidar_poses(poses_filepath: str, lidar_filepath: str, pose_time_tolerance: float = 1.):
    # Read global poses from .csv file and link each lidar_scan with the nearest pose
    # threshold: threshold in seconds
    # Returns a dictionary with (4, 4) pose matrix indexed by a timestamp (as integer)

    with open(poses_filepath, "r") as h:
        txt_poses = h.readlines()

    n = len(txt_poses)
    system_timestamps = np.zeros((n,), dtype=np.int64)
    poses = np.zeros((n, 4, 4), dtype=np.float64)       # 4x4 pose matrix

    for ndx, pose in enumerate(txt_poses):
        # Split by comma and remove whitespaces
        temp = [e.strip() for e in pose.split(',')]
        assert len(temp) == 13, f'Invalid line in global poses file: {temp}'
        system_timestamps[ndx] = int(temp[0])
        poses[ndx] = np.array([[float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4])],
                               [float(temp[5]), float(temp[6]), float(temp[7]), float(temp[8])],
                               [float(temp[9]), float(temp[10]), float(temp[11]), float(temp[12])],
                               [0., 0., 0., 1.]])

    # Ensure timestamps and poses are sorted in ascending order
    sorted_ndx = np.argsort(system_timestamps, axis=0)
    system_timestamps = system_timestamps[sorted_ndx]
    poses = poses[sorted_ndx]

    # List LiDAR scan timestamps
    all_lidar_timestamps = [int(os.path.splitext(f)[0]) for f in os.listdir(lidar_filepath) if
                            os.path.splitext(f)[1] == '.bin']
    all_lidar_timestamps.sort()

    lidar_timestamps = []
    lidar_poses = []
    count_rejected = 0

    for ndx, lidar_ts in enumerate(all_lidar_timestamps):
        # Skip faulty point clouds
        if lidar_ts in FAULTY_POINTCLOUDS:
            continue

        # Find index of the closest timestamp
        closest_ts_ndx = find_nearest_ndx(lidar_ts, system_timestamps)
        delta = abs(system_timestamps[closest_ts_ndx] - lidar_ts)
        # Timestamp is in nanoseconds = 1e-9 second
        if delta > pose_time_tolerance * 1000000000:
            # Reject point cloud without corresponding pose
            count_rejected += 1
            continue

        lidar_timestamps.append(lidar_ts)
        lidar_poses.append(poses[closest_ts_ndx])

    lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)
    lidar_poses = np.array(lidar_poses, dtype=np.float64)     # (northing, easting) position

    print(f'{len(lidar_timestamps)} scans with valid pose, {count_rejected} rejected due to unknown pose')
    return lidar_timestamps, lidar_poses


def relative_pose(m1, m2):
    # SE(3) pose is 4x 4 matrix, such that
    # Pw = [R | T] @ [P]
    #      [0 | 1]   [1]
    # where Pw are coordinates in the world reference frame and P are coordinates in the camera frame
    # m1: coords in camera/lidar1 reference frame -> world coordinate frame
    # m2: coords in camera/lidar2 coords -> world coordinate frame
    # returns: relative pose of the first camera with respect to the second camera
    #          transformation matrix to convert coords in camera/lidar1 reference frame to coords in
    #          camera/lidar2 reference frame
    #
    m = np.linalg.inv(m2) @ m1
    # !!!!!!!!!! Fix for relative pose !!!!!!!!!!!!!
    m[:3, 3] = -m[:3, 3]
    return m
