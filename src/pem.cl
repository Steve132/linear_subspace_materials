#define BRDF_SAMPLING_RES_THETA_H       90
#define BRDF_SAMPLING_RES_THETA_D       90
#define BRDF_SAMPLING_RES_PHI_D         360

#define RED_SCALE (1.0/1500.0)
#define GREEN_SCALE (1.15/1500.0)
#define BLUE_SCALE (1.66/1500.0)
#ifndef M_PI
#define M_PI	3.1415926535897932384626433832795
#endif

// rotate vector along one axis
static inline float3 rotate_vector(float3 vector, float3 axis, float2 sc_angle)
{
	float3 out=vector*sc_angle.y;

	float temp = dot(axis,vector);
	temp = temp*(1.0-sc_angle.y);

	out+= axis*temp;

	float3 cresult=cross(axis,vector);
	
	out+=cresult*sc_angle.x;
	return out;
}

struct halfdiff
{
	float2 tp_half;
	float3 tp_diff;
};

// convert standard coordinates to half vector/difference vector coordinates
static inline struct halfdiff std_coords_to_half_diff_coords(float3 w_in,float3 w_out)
{

	float3 w_half = normalize((w_in + w_out)/2.0);
	
	struct halfdiff out;
	// compute  theta_half, fi_half
	out.tp_half.x=acos(w_half.z);
	out.tp_half.y=atan2(w_half.y,w_half.x);

	// compute diff vector
	float3 temp=rotate_vector(w_in, (float3)(0.0,0.0,1.0) , -out.tp_half.y);
	float3 diff=rotate_vector(temp, (float3)(0.0,1.0,0.0) , -out.tp_half.x);
	
	// compute  theta_diff, fi_diff	
	out.tp_diff.x = acos(diff.z);
	out.tp_diff.y = atan2(diff.y, diff.x);
	return out
}

// Lookup theta_half index
// This is a non-linear mapping!
// In:  [0 .. pi/2]
// Out: [0 .. 89]
static inline unsigned int theta_half_index(float theta_half)
{
	if (theta_half <= 0.0)
		return 0;
	float theta_half_deg = ((theta_half / (M_PI/2.0))*BRDF_SAMPLING_RES_THETA_H);
	float temp = theta_half_deg*BRDF_SAMPLING_RES_THETA_H;
	temp = sqrt(temp);
	int ret_val = (int)temp;
	if (ret_val < 0) ret_val = 0;
	if (ret_val >= BRDF_SAMPLING_RES_THETA_H)
		ret_val = BRDF_SAMPLING_RES_THETA_H-1;
	return ret_val;
}


// Lookup theta_diff index
// In:  [0 .. pi/2]
// Out: [0 .. 89]
static inline unsigned int theta_diff_index(float theta_diff)
{
	int tmp = int(theta_diff / (M_PI * 0.5) * BRDF_SAMPLING_RES_THETA_D);
	if (tmp < 0)
		return 0;
	else if (tmp < BRDF_SAMPLING_RES_THETA_D - 1)
		return tmp;
	else
		return BRDF_SAMPLING_RES_THETA_D - 1;
}


// Lookup phi_diff index
static inline unsigned int phi_diff_index(float phi_diff)
{
	// Because of reciprocity, the BRDF is unchanged under
	// phi_diff -> phi_diff + M_PI
	if (phi_diff < 0.0)
		phi_diff += M_PI;

	// In: phi_diff in [0 .. pi]
	// Out: tmp in [0 .. 179]
	int tmp = int(phi_diff / M_PI * BRDF_SAMPLING_RES_PHI_D / 2);
	if (tmp < 0)	
		return 0;
	else if (tmp < BRDF_SAMPLING_RES_PHI_D / 2 - 1)
		return tmp;
	else
		return BRDF_SAMPLING_RES_PHI_D / 2 - 1;
}

static inline float3 lookup_brdf_val(float3 w_in,float3 w_out,const global float* brdf)
{
	struct halfdiff hd=std_coords_to_half_diff_coords(w_in,w_out);
	unsigned int ind=phi_diff_index(hd.tp_diff.y)
			+theta_diff_index(hd.tp_diff.x)*BRDF_SAMPLING_RES_PHI_D/2
			+theta_half_index(hp.tp_half.x)*BRDF_SAMPLING_RES_PHI_D/2 * (BRDF_SAMPLING_RES_THETA_D);

	float red_val = brdf[ind] * RED_SCALE;
	float green_val = brdf[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D/2] * GREEN_SCALE;
	float blue_val = brdf[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D] * BLUE_SCALE;

	return (float3)(red_val,green_val,blue_val);
}
struct rotmat
{
	float3 u,v,w;
};
static inline struct rotmat rotZtoTarget(float3 target,float3 up)
{
	struct rotmat rm;
	rm.w=target;
	rm.v=normalize(cross(target,up));
	rm.u=normalize(cross(rm.w,rm.v));
	returm rm;
}

const sampler_t envsampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;

static inline float3 cubemap_lookup(image2d_t envmap,float3 vec)
{
	float3 avec=abs(vec);

	unsigned int ma= (avec.x > avec.y) ? 0 : ((avec.y > avec.z) ? 1 : 2);
	uint3 mask = (uint3)((ma+1) % 3,(ma+2) % 3, ma);
	float3 f=shuffle(vec,mask);
	f.x/=f.z;f.y/=f.z;
	
	ma+=1;
	ma=(f.z < 0.0) ? -ma : ma; 
	
	static const float2 offsets[7]=
		{
		(float2)(2.0,3.0),//-3
		(float2)(2.0,2.0),//-2
		(float2)(1.0,1.0),//-1
		(float2)(1.0,0.0),//0 NULL
		(float2)(3.0,1.0),
		(float2)(2.0,2.0),
		(float2)(2.0,1.0),
		};
	float2 uv=offsets[ma+3]/4.0;
	return read_imagef(envmap,envsampler,uv).xyz;
}

__kernel void prefilter_environment_map_merl(
	global double3* pem_buffer,
	read_only image2d_t envmap,
	const global double* brdf,
	const globat float3* w_out_from_index,
	const global float3* r_target_from_index,
	const unsigned num_samples_so_far,
	const unsigned num_samples,
	const __constant float2* uniformrandom)
{
	double3 curval=0.0;//theta goes vertical, phi goes all around

	//w_out,r_target;
	size_t w_out_id=get_global_id(0);
	size_t r_target_id=get_global_id(1);

	float3 w_out=w_out_from_index[w_out_id];
	float3 target_r=r_target_from_index[r_target_id];
	
	struct rotmat rm=rotZtoTarget(target_r,(float3)(0.0,0.0,1.0));

	for(unsigned int s_i=0;s_i < num_samples;si++)
	{
		float2 samplepoint=uniformrandom[s_i]*(float2)(M_PI/2.0,2.0*M_PI);
		double quadrature=sin(samplepoint.x);
		float psca=quadrature;
		float3 w_in=(float3)(psca*cos(samplepoint.y),psca*sin(samplepoint.y),cos(samplepoint.x));
		float3 f=lookup_brdf_val(w_in,w_out,brdf)*w_in.z;
		float3 new_win=(float3)(dot(rm.u,w_in),dot(rm.v,w_in),dot(rm.w,w_in));
		f*=cubemap_lookup(envmap,new_win)/quadrature;
		curval+=f;
	}	
	double new_normalization=num_samples_so_far+num_samples;
	pem_buffer[pem_index]=(pem_buffer[pem_index]*num_samples_so_far+curval)/(new_normalization);
}	
