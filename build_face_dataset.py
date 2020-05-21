from imutils.video import VideoStream
import argparse
import imutils
import time
import os
import cv2


ap = argparse.ArgumentParser()
#  specify both shorthand -c and longhand -- cascade version
ap.add_argument("-c", "--cascade", required=True, help="path to face cascade")
ap.add_argument("-o", "--output", required=True, help="path to output directory")
# parse_args is called with no argument and ap will automatically determine the command line arguments
args = vars(ap.parse_args())  # vars return the __dict__ attribute of the object

# load cv's Haar cascade for face detection
detector = cv2.CascadeClassifier(args["cascade"])

# initialize video stream
print("starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2)  # warm up camera sensor
total = 0  # number of face images stored

# loop over frames from video stream
while True:
    # get the frame
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)
    # image: gray scale
    # scaleFactor:  how much image size is reduced at each scale
    # minNeighbour: how many neighbour should each bounding box have to be considered valid detection
    # minSize:      minimum possible face image size
    rects = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    # loop over the rects and draw on frame
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # press 'k' to write the frame to the disk
    print("running")
    if key == ord('k'):
        print("k")
        p = os.path.sep.join([args["output"], "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, orig)
        total += 1
    # press 'q' to quit loop
    elif key == ord('q'):
        print("q")
        break

print(f"{total} faces are stored")
cv2.destroyAllWindows()  # destroy all the windows created
vs.stop()