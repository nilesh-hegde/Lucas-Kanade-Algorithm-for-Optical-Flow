# Import modules needed for the script
import sys
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import convolve2d as conv2
from PIL import Image
import random
import skimage.morphology
from skimage.io import imread_collection


'''
Arguments : -> image - image whose gradients need to be calculated
			
Returns : -> Ix - X-gradient of image calculated using defined kernel
		  -> Iy - Y-gradient of image calculated using defined kernel

Description : This function return the gradients in x-direction and y-direction
'''
def ImageGradient(image):
	# Define the kernel to find gradients
    kernel = np.array([[-1, 1],[-1, 1]])
    
    # Convolve the image with the given kernel
    Ix = conv2(image,kernel,boundary='symm',mode='same')
    Iy = conv2(image,kernel.T,boundary='symm',mode='same')
    
    # Return the convolved images
    return Ix,Iy


'''
Arguments : -> image - image whose gradients need to be calculated
			
Returns : -> Is - Smoothened image calculated using defined kernel

Description : This function return the gradients in x-direction and y-direction
'''
def ImageSmoothening(image,size = 2):
	# Define kernel for smoothening
    kernel = np.ones((size,size))/(size**2)
    
    # Convolve the image with the given kernel
    Is = conv2(image,kernel,boundary='symm',mode='same')
    
    # return the convolved image
    return Is


'''
Arguments : -> frame1 - first frame used to find oprical flow
			-> frame2 - second frame used to find oprical flow
			-> window_size - size of kernel used in smoothening
			-> num_features - maximum number of features to display
			-> t - threshold used in plotting arrows
			-> mode - specifies whether dense optical flow needs to be found or sparse
			
Returns : -> u,v - store flow vectors
		  -> feature-map - contains good features to track
		  
Description : This function implements Lucas-Kanade algorithm for both sparse and dense optcal flow calculations.
'''
def LucasKanade(frame1, frame2, window_size = 3, num_features = 10000, t=1):
	# Convert frames to numpy array
    frame1 = np.array(frame1)
    frame2 = np.array(frame2)
    
    # Initialize feature map
    feature_map = np.zeros(frame1.shape)
    
	# Find gradients
    Ix,Iy = ImageGradient(frame1)
    
    # Find difference after smootheneing
    It = ImageSmoothening(frame2) - ImageSmoothening(frame1)
    
    # Create vectors to hold vector flow
    u = np.zeros(frame1.shape)
    v = np.zeros(frame1.shape)
    feature_map = np.zeros(frame1.shape)
    
    # Find good features to track
    features = cv2.goodFeaturesToTrack(frame1, 10000 ,0.01 ,10).reshape(-1,2)
    
    # Update feature map
    for f in range(features.shape[0]):
        j,i = features[f,0] , features[f,1]
        i = np.int(i)
        j = np.int(j)
        feature_map[i,j]=1
    
    # Find vector flow
    for i in range(3,frame1.shape[0]-3):
        for j in range(3,frame1.shape[1]-3):
            Wx = np.array([Ix[i-1,j-1],Ix[i,j-1],Ix[i+1,j-1],Ix[i-1,j],Ix[i,j],Ix[i+1,j],Ix[i-1,j+1],Ix[i,j+1],Ix[i+1,j+1]])
            Wy = np.array([Iy[i-1,j-1],Iy[i,j-1],Iy[i+1,j-1],Iy[i-1,j],Iy[i,j],Iy[i+1,j],Iy[i-1,j+1],Iy[i,j+1],Iy[i+1,j+1]])
            Wt = np.array([It[i-1,j-1],It[i,j-1],It[i+1,j-1],It[i-1,j],It[i,j],It[i+1,j],It[i-1,j+1],It[i,j+1],It[i+1,j+1]])
				
				# Stack Wx, Wy and solve (AA^T)^{-1}A^Tb to get vector flow
            A = np.vstack((Wx,Wy)).T
            b = np.dot(A.T,Wt)
            A = np.dot(A.T,A)
            A_inv = np.linalg.pinv(A)
            (u[i,j],v[i,j]) = np.dot(A_inv,b)
    
    # Plot the vector flow in first frame
    fig, ax = plt.subplots(figsize = (12, 12))
    ax.set_title("Vector Field")
    ax.imshow(frame1, cmap = "gray")
    for i in range(frame1.shape[0]):
        for j in range(frame1.shape[1]):
            if feature_map[i,j] == 1:
                if abs(u[i,j])>t or abs(v[i,j])>t:
                    plt.arrow(j,i,v[i,j],u[i,j],head_width = 5, head_length = 5, color = "y")
    plt.axis('off')
    plt.close('all')
    
    # Return vectors and feature map
    return u,v,feature_map


'''
Arguments : -> u,v - Vectors containing details on vector flow
			-> feature_map - Contains locations of features
			-> frame1 - First frame used to find oprical flow
			-> idx - Frame number
			-> t - threshold used in plotting arrows
			-> thresh - Threshold for masking
			
Returns : None
		  
Description : This function implements object detection using the flowvector and feature maps obtained by Lucas-Kanade algorithm
'''
def Detect(u,v,feature_map,frame1,idx,t=0.2,thresh=0.8):
	# Convert frame to numy array
    frame1 = np.array(frame1)
    
    # Create segmentation mask
    seg_mask=((u*u + v*v)>thresh)
    
    # Create a morphological disk
    morph_disk = skimage.morphology.disk(5)
    
    # Compute convex hull
    seg_mask = skimage.morphology.convex_hull_object(seg_mask)
    
    # Remove small foreground objects 
    seg_mask = skimage.morphology.binary_opening(seg_mask, morph_disk)
    
    # Remove small connected components
    seg_mask = skimage.morphology.remove_small_objects(seg_mask, 5000)
    
    # Create plot for image with bounding box
    fig,ax3 = plt.subplots(figsize=(12,12))
    ax3.set_title("Optical Flow with detection:")
    obj_mask = skimage.measure.label(seg_mask)
    ax3.imshow(frame1,cmap='gray')
    
    # Draw arrow for flow vector
    for i in range(frame1.shape[0]):
        for j in range(frame1.shape[1]):
            if feature_map[i,j] == 1:
                if abs(u[i,j])>t or abs(v[i,j])>t:
                    ax3.arrow(j,i,v[i,j],u[i,j],head_width = 5, head_length = 5, color = "y")
    
    # Draw bounding boxes
    for region in skimage.measure.regionprops(obj_mask):
        y, x, y2, x2 = region.bbox
        if(np.sum(feature_map[y:y2,x:x2])!=0) :
            height = y2 - y
            width = x2 - x
            ax3.add_patch(patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='g',facecolor='none'))
    plt.axis('off')
    
    # Save image
    plt.savefig('./Output/output_'+str(idx)+'.jpg')
    plt.close('all')


'''
Arguments : -> frame_folder - folder in which frames whose optical flow needs to be calculated are located
			
Returns : None
		  
Description : This function implements object detection using the flowvector and feature maps obtained by Lucas-Kanade algorithm for all frames in a folder
'''
def gen_bboxes_vid(frame_folder):
    frames = imread_collection(frame_folder + '/*.pgm')
    for i in range(1,len(frames)):
        u,v,feature_map = LucasKanade(frames[i],frames[i-1],10000,t=0.2)
        Detect(u,v,feature_map,frames[i],idx = i,thresh=0.5)


'''
Arguments : None

Returns : None

Description : This function is the start of execution for the script. It acceses all functions defined above to seamlessly find sparse optical flow.
'''
def main():
	# Create folder 'Output' if it does not exist
	if not os.path.exists('Output'):
		os.makedirs('Output')
		print("Folder created")
	
	# Generate output frames
	gen_bboxes_vid(sys.argv[1])

if __name__ == "__main__":
    main()





