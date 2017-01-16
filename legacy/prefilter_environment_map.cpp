#include<iostream>
#include<Eigen/Dense>

Eigen::Matrix3d rot_from_upvector(const Eigen::Vector3d& lookat,const Eigen::Vector3d& upvector)
