# Import modules needed for the script
import cv2
import os
import sys


'''
Arguments : -> video_path - Relative path to the videos whose frames needs to be extarcted
			-> save_path - Relative path to where folder of saved frames need to be located at
Returns : None

Description : This function converts a video into frames of .pgm type.
'''
def VideoToFrames(video_path,save_path = ''):
    # Initialize video capture object
    cap = cv2.VideoCapture(video_path)
    
    # Create a folder to save frames to
    if(save_path == ''):
        outFolder = video_path.split('.')[0] + '_frames'
    else:
        outFolder = save_path
    if (os.path.exists(outFolder)!= 1):
        os.mkdir(outFolder)
    
    cnt = 0
    
    # Save a maximum of 400 frames in greyscale
    while(cap.isOpened() and cnt<400):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if(ret):
            cv2.imwrite(outFolder + '/frame%d.pgm' %cnt,frame)
            cnt+=1
        else:
            break
 
    cap.release() 


'''
Arguments : None

Returns : None

Description : This function is the start of execution for the script. It acceses all functions defined above to seamlessly extract frames.
'''
def main():
	VideoToFrames(sys.argv[1])

if __name__ == "__main__":
    main()

