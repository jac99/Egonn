# EgonNN Egocentric Neural Network for Point Cloud Based 6DoF Relocalization at the City Scale

Paper: [Egocentric Neural Network for Point Cloud Based 6DoF Relocalization at the City Scale](http://arxiv.org/xxxxxxx) 

[Jacek Komorowski](mailto:jacek.komorowski@pw.edu.pl), Monika Wysoczanska, Tomasz Trzcinski

Warsaw University of Technology

### Introduction
The paper presents a deep neural network-based method for global and local descriptors extraction from a point cloud acquired by a rotating 3D LiDAR sensor.
The descriptors can be used for two-stage 6DoF relocalization. First, a course position is retrieved by finding candidates with the closest global descriptor in the database of geo-tagged point clouds. Then, 6DoF pose between a query point cloud and a database point cloud is estimated by matching local descriptors and using a robust estimator such as RANSAC.
Our method has a simple, fully convolutional architecture and uses a sparse voxelized representation of the input point cloud. It can efficiently extract a global descriptor and a set of local keypoints in a single pass through the network.

## Training/evaluation code and pre-trained models will be released after the paper acceptance.
