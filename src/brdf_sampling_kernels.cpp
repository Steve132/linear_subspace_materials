#include<random>
#include<Eigen/Dense>


class BRDF
{
public:
	virtual Eigen::Vector3d sample(const Eigen::Vector3f& w_in,const Eigen::Vector3f& w_out) const=0;
	~BRDF()
	{}
};

struct sampling_result
{
	std::vector<Eigen::Matrix3d> colors;
	std::vector<Eigen::Matrix3d> sample_directions;
};
std::vector<Eigen::Matrix9d> sample_n(const std::vector<Eigen::Vector3f>& w_out
