import os
import numpy as np
import open3d as o3d

from datasets.southbay.southbay_raw import SouthBayDataset
from datasets.southbay.generate_training_tuples import generate_triplets
from datasets.southbay.southbay_raw import load_pc

data_root = '/media/sf_Datasets/ApolloSouthBay'

#pcd = o3d.io.read_point_cloud(pc_path)
#pc = np.asarray(pcd.points)
#print("output array from input list : ", pc)

ds = SouthBayDataset(data_root)
print(ds.location_ndx)

ds.print_info()

path = ds.global_ndx[0].rel_scan_filepath
path = os.path.join(data_root, path)
pc = load_pc(path)
print('.')

#generate_triplets(ds, 'MapData', 'TrainData')

