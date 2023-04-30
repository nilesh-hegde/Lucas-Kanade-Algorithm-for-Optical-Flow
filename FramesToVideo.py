# Import modules needed for the script
import sys
import cv2
import os


'''
Arguments : -> path - Relative or absolute path to folder containing all frames
			
Returns : None

Description : This function converts frames into video
'''
def FramesToVideo(path,fps):
	# Get all image filenames in the directory
	filenames = [f for f in os.listdir(path) if f.endswith('.jpg')]

	# Load the first image to get the image size
	img = cv2.imread(os.path.join(path, filenames[0]))
	height, width, channels = img.shape

	# Define the video codec and create a VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

	# Loop through each image and write it to the video
	for filename in filenames:
		img = cv2.imread(os.path.join(path, filename))
		out.write(img)

	# Release the VideoWriter and close all windows
	out.release()
	cv2.destroyAllWindows()
	
'''
Arguments : None

Returns : None

Description : This function is the start of execution for the script. It acceses all functions defined above to seamlessly create a video.
'''
def main():
	FramesToVideo(sys.argv[1], int(sys.argv[2]))

if __name__ == "__main__":
    main()



