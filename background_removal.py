import cv2
import numpy as np
import sys




detector = cv2.SimpleBlobDetector()
def main():
    #training_img = cv2.imread('disc_orb_2.jpg')
    cx = 0
    cy = 0
    centers = np.array([[0],[0]])
    print(centers)
    frameCount = 0
    #bgr_train = training_img
    #trainKeypoints, trainDescriptors = orb.detectAndCompute(bgr_train, None)


    frisbeeVid = cv2.VideoCapture('blue_above.mp4')
    ret, frame = frisbeeVid.read()

    height, width, layers = frame.shape
    size = (width, height)
    out = cv2.VideoWriter('fp.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

    if not ret:
        print("video failed to load")
        sys.exit()  # break in case i messed up'

    vidHeight = frame.shape[0]
    vidWidth = frame.shape[1] #establishing video size
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    newVid = cv2.VideoWriter("frisbee_track.avi", fourcc=fourcc, fps=24.0,
                            frameSize=(1684, 1088))  # set up new video output
    frisbeeVid.set(1, 0)  # reset to frame 0


    backSub = cv2.createBackgroundSubtractorMOG2(history=7, varThreshold=60, detectShadows=False);

    while (frisbeeVid.isOpened()):

        ret, frame = frisbeeVid.read()
        if not ret:
            print("you messed up in the while loop, or the video ended not sure lol")
            print(frameCount)
            break

        blur = cv2.GaussianBlur(frame, (7, 7), 0)

        fgmask = backSub.apply(blur);
        fgmask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        ret, binary = cv2.threshold(fgmask, 100, 255,
                                    cv2.THRESH_OTSU)

        #thresh = cv2.erode(binary,None,iterations = 2)
        thresh = cv2.erode(binary, None, iterations=2)
        #thresh = cv2.dilate(thresh, None, iterations=4)
        #cv2.imshow('thresh test',thresh)
        #cv2.waitKey(0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours (in red) on the original image and display the result
        # Input color code is in BGR (blue, green, red) format
        # -1 means to draw all contours
        if contours is not None:
            with_contours = cv2.drawContours(frame, contours, -1, (255, 0, 255), 3)
            count = 0
            cx = 0
            cy = 0
            for i in contours:
                M = cv2.moments(i)
                if M['m00'] != 0:
                    cx = cx + M['m10'] / M['m00']
                    cy = cy + M['m01'] / M['m00']
                    count = count+1
            if count != 0:
                cx = int(cx / count)
                cy = int(cy / count)
                #print("cx",cx,"\ncy",cy)
                newCenter = np.array([[cx], [cy]])
                #print(newCenter)
                centers = np.append(centers, newCenter, axis=1)
                #print("Centers:",centers)
            for j in range(1,centers.shape[1]):
                cv2.circle(with_contours, (centers[0][j],centers[1][j]), 7, (255,255,255), 1)







        #cv2.imshow('blur', fgmask_bgr)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Get the video
        out.write(with_contours)
        frameCount = frameCount + 1


    out.release()




if __name__ == "__main__":
    main()