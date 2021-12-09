import cv2
import numpy as np
import math

#Turn on Magnilink MacViewer if camera view doesn't appear!
# 0 = Magnilink cam (if connected, otherwise webcam)
cap = cv2.VideoCapture("/Users/maxborglowe/Desktop/IEA LTH/Examensarbete EITL05 LVI/Video/shake_test1.mov")

# Text definition stuff
font                   = cv2.FONT_HERSHEY_DUPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

# Camera specs
FPS = 30
frame_time = round(1/FPS, 5)

# Measured object dimensions, input all external values here!!!
# Note: all length values are declared in millimeters
object_w_actual = 52    #actual object width
object_w_cap = 130      #on-screen measurement of object width
cam_h = 200             #height of camera (lens)
ppi = 227               #given ppi for used computer monitor
# End of inputs. Don't edit anything else!!!


zoom = object_w_cap/object_w_actual     #camera zoom based on object measurement, actual and on-screen
in_to_mm = 25.4                         #convert inches to mm
ppmm_screen = ppi/in_to_mm              #pixel/mm on 13" macbook pro 2015
ppmm_real = ppmm_screen/zoom            #actual length of object on screen

max_speed = 0                           #maximum velocity [mm/s]
ang_vel = 0                             #angular velocity

amtCorners = 0                          #corners detected in image
array_len = 2                           #used for mean calculation of object coordinates
array_p = 0                             #array pointer

xMean = [0]*array_len
yMean = [0]*array_len
xMeanAbsolute = 0
yMeanAbsolute = 0

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   #screen width
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  #screen height
rect_w = 150                            #bounding rectangle width, used during corner detection
rect_h = 150                            #bounding rectangle height
rect_x = int((w-rect_w)/2)
rect_y = int((h-rect_h)/2)

wait_for_stable = 0
stable_limit = 10

while (cap.isOpened()):
    ret, frame = cap.read()

    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    gray1 = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray1, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = np.zeros(thresh1.shape, np.uint8)
    mask[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = gray1[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]
    thresh1 = cv2.bitwise_and(thresh1, thresh1, mask = mask)  

    gray = np.float32(thresh1)

    corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 3)
    if corners is not None:
        corners = np.int0(corners)

    xMean[array_p] = 0
    yMean[array_p] = 0
    amtCorners = 0

    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (x,y), 2, (0, 200, 255), -1)
            xMean[array_p] += x
            yMean[array_p] += y
            amtCorners += 1
    
    if array_len > 1 and amtCorners != 0:
        xMean[array_p] /= (amtCorners)
        yMean[array_p] /= (amtCorners)
    array_p = (array_p + 1)%array_len

    speed_ref_p = 0
    speed_end_p = array_len - 1

    #calculate speed in mm/s
    speed = round((math.sqrt(pow(xMean[speed_end_p]-xMean[speed_ref_p], 2)+pow(yMean[speed_end_p]-yMean[speed_ref_p], 2))/((array_len-1)*frame_time))/ppmm_real, 1)

    if speed < 3000 and speed >= max_speed: #filter unreasonably high speeds
        max_speed = speed
    
    ang_vel = round(math.degrees(math.atan(max_speed/cam_h)), 2)
    rpm = 0
    if ang_vel > 0:
        rpm = round(ang_vel/0.1666667) #60 sec / 360˚ = 0.1666667

    for i in xMean:
        xMeanAbsolute += i
    for i in yMean:
        yMeanAbsolute += i
        
    xMeanAbsolute /= (array_len + 1)
    yMeanAbsolute /= (array_len + 1)
    xMeanAbsolute = int(xMeanAbsolute)
    yMeanAbsolute = int(yMeanAbsolute)

    if xMeanAbsolute < 5 and yMeanAbsolute < 5:
        rect_x = int((w-rect_w)/2)
        rect_y = int((h-rect_h)/2)
        wait_for_stable = 0

    # boundaries follow green dot
    if stable_limit > 0 and wait_for_stable > stable_limit and xMeanAbsolute >= 5 and yMeanAbsolute >= 5:
        rect_x = int(xMeanAbsolute-rect_w/2)
        rect_y = int(yMeanAbsolute-rect_h/2)
    
    wait_for_stable += 1

    #print("array_p = " + str(array_p) + " x: " + str(xMeanAbsolute) + " y: " + str(yMeanAbsolute))

    #paint the object follower dot
    cv2.circle(frame, (int(xMeanAbsolute), int(yMeanAbsolute)), 12, (0, 255, 0), -1)
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x+rect_w, rect_y+rect_h), (255, 255, 0), 2, -1)

    cv2.putText(frame, "x: " + str(xMeanAbsolute) + " y: " + str(yMeanAbsolute) + " speed: " + str(speed) + " mm/s, max speed: " + str(max_speed),
    (10, 600), 
    font, 
    fontScale,
    fontColor,
    lineType)

    cv2.putText(frame, "angular velocity: " + str(ang_vel) + " deg/s --> rpm: " + str(rpm),
    (10, 700), 
    font, 
    fontScale,
    fontColor,
    lineType)

    cv2.putText(frame, "window width: " + str(w) + " window height: " + str(h),
    (10, 30), 
    font, 
    fontScale,
    fontColor,
    lineType)

    cv2.imshow('dst', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()