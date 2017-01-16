import numpy as np
import pyopencl as cl
import imageio
import argparse
from matplotlib import pyplot as plt

#hemisphere size to control outgoing direction in rotated space: (4 n^2/2+n^2)=3*n^2
#number of isotropic tangent spaces=6*m^2
#

def gen_triangle_1d(topimg,bottomimg):
	n=topimg.shape[0]
	tout=np.triu(topimg)
	tout+=np.tril(bottomimg)
	tout+=(np.diag(np.diag(topimg))+np.diag(np.diag(bottomimg)))/2.0
	return tout

def gen_triangle(t,b):
	if(len(t.shape) > 2):
		return np.dstack([gen_triangle_1d(t[:,:,i],b[:,:,i]) for i in range(t.shape[2])])
	else:
		return gen_triangle_1d(t,b)
	

def vertical_cubemap_to_blurrable_cubemap(cm):
	
	n=cm.shape[0]/4

	cms=list(cm.shape)
	cms[1]+=n
	blurrablecm=np.zeros(cms)
	blurrablecm[:,n:,:]=cm
	
	top=cm[0:n,n:(2*n),:]
	left=cm[n:(2*n),0:n,:]
	front=cm[n:(2*n),n:(2*n),:]
	right=cm[n:(2*n),(2*n):(3*n),:]
	bottom=cm[(2*n):(3*n),n:(2*n),:]
	back=cm[(3*n):(4*n),n:(2*n),:]

	ul=gen_triangle(np.rot90(left,3),np.rot90(top))
	blurrablecm[0:n,n:(2*n),:]=ul

	ur=np.rot90(gen_triangle(right,np.rot90(top,2)))
	blurrablecm[0:n,(3*n):(4*n),:]=ur

	mr=gen_triangle(np.rot90(bottom),np.rot90(right,3))
	blurrablecm[(2*n):(3*n),(3*n):(4*n),:]=mr

	ml=np.rot90(gen_triangle(np.rot90(bottom,2),left))
	blurrablecm[(2*n):(3*n),n:(2*n),:]=ml

	ll=np.rot90(gen_triangle(np.rot90(back,2),np.rot90(left,1)))
	blurrablecm[(3*n):(4*n),n:(2*n),:]=ll

	lr=gen_triangle(np.rot90(back,1),np.rot90(right,2))
	blurrablecm[(3*n):(4*n),(3*n):(4*n),:]=lr

	blurrablecm[n:(2*n),0:n,:]=np.rot90(back,2)

	blurrablecm[0:n,0:n,:]=np.rot90(top,2)
	blurrablecm[(2*n):(3*n),0:n,:]=np.rot90(bottom,2)
	blurrablecm[(3*n):(4*n),0:n,:]=np.rot90(front,2)
	
	plt.imshow(np.sqrt(blurrablecm))
	plt.show()

	return blurrablecm

def prefilter_merl(bcm,w_out_dm,tangents_normals_dim,merlbrdfdata):
	outbuffer=np.zeros((tangents_normals_dim,w_out_dim,3))
	clbcm=cl.image_from_array(ctx,bcm.astype(np.float32),num_channels=bcm.shape[2])


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Prefilter an environment map with an isotropic brdf')
	parser.add_argument('--environment','-e',required=True,help='environment map cross file')
	brdf_group=parser.add_mutually_exclusive_group(required=True)
	brdf_group.add_argument('--merlfile', type=argparse.FileType('rb'),help='merl data file')
	brdf_group.add_argument('--diffuse',help='diffuse color')
	parser.add_argument('--w_out_dim','-m',type=int,default=32,help='w_out cubemap resolution')
	parser.add_argument('--tangents_normals_dim','-n',type=int,default=32,help='cubemap resolution tangents')
	parser.add_argument('--outfile','-o',type=argparse.FileType('wb'),help='output prefiltered environment map')
	args=parser.parse_args()
	
	cm=imageio.imread(args.environment)
	cm=vertical_cubemap_to_blurrable_cubemap(cm)
	
