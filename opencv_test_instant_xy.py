import cv2
import numpy as np
import math

#Set LiveRec to true video input is live
liveRec = False

if(liveRec is False):
    #replace string within "" with the directory of the video you wish to measure
    cap = cv2.VideoCapture("/Users/maxborglowe/Desktop/IEA LTH/Examensarbete EITL05 LVI/Video/shake_test3.mov")
else:
    #Turn on Magnilink MacViewer if camera view doesn't appear!
    # 0 = Magnilink cam if connected, otherwise webcam
    cap = cv2.VideoCapture(0)

# Text definition stuff
font                   = cv2.FONT_HERSHEY_DUPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

fontColorRPM           = (0,255,255)

# Camera specs
FPS = 30
frame_time = round(1/FPS, 7)

# Measured object dimensions, input all external values here!!!
# Note: all length values are declared in millimeters
object_w_actual = 1.1    #actual object width [mm]
object_w_cap = 12.0      #on-screen measurement of object width [mm]
cam_h = 305             #height of camera (lens) [mm]
ppi = 227               #given ppi [pixels/inch] for used computer monitor
# End of inputs. Don't edit anything else!!!

zoom = object_w_cap/object_w_actual     #camera zoom based on object measurement, actual and on-screen
in_to_mm = 25.4                         #convert inches to mm
ppmm_screen = ppi/in_to_mm              #pixel/mm on 13" macbook pro 2015
ppmm_real = ppmm_screen/zoom            #actual length of object on screen

max_speed = [0, 0]                           #maximum velocity [mm/s]
ang_vel =[0, 0]                             #angular velocity

amtCorners = 0                          #corners detected in image
array_len = 2                           #used for mean calculation of object coordinates
array_p = 0                             #array pointer

xMean = 0                               #mean x-coordinate based on corners found in video
xMean_prev = 0
yMean = 0                               #mean y-coordinate based on corners found in video
yMean_prev = 0

speed_c = 0                             #counter for avg coordinate measurement
speed_mod = 2                           #counter cutoff

speed = [0, 0]                          #RPM


w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   #screen width
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  #screen height
rect_w = 300                            #bounding rectangle width, used during corner detection
rect_h = 300                            #bounding rectangle height
rect_x = int((w-rect_w)/2)
rect_y = int((h-rect_h)/2)

frame_size = (int(cap.get(3)), int(cap.get(4)))
out = cv2.VideoWriter('/Users/maxborglowe/Desktop/outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), FPS, frame_size)

wait_for_stable = 0                     #bounding box stability counter
stable_limit = 10                       #stability criteria

while (cap.isOpened()):
    ret, frame = cap.read()
    
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    gray1 = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray1, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = np.zeros(thresh1.shape, np.uint8)
    mask[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = gray1[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]
    thresh1 = cv2.bitwise_and(thresh1, thresh1, mask = mask)  

    gray = np.float32(thresh1)

    corners = cv2.goodFeaturesToTrack(gray, 300, 0.01, 1)
    if corners is not None:
        corners = np.int0(corners)

    xMean = 0
    yMean = 0
    amtCorners = 0

    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            #paint yellow dots that show the effects of edge detection performed on the video
            cv2.circle(frame, (x,y), 1, (0, 200, 255), -1)
            xMean += x
            yMean += y
            amtCorners += 1
    
    if amtCorners != 0:
        xMean /= amtCorners
        yMean /= amtCorners

    xMean = int(xMean)
    yMean = int(yMean)

    #store previous coordinate
    if speed_c == 0:
        xMean_prev = xMean
        yMean_prev = yMean
    
    speed_c += (speed_c+1)%speed_mod
    #calculate speed in mm/s
    speed[0] = round(((xMean-xMean_prev)/frame_time)/ppmm_real)
    speed[1] = round(((yMean-yMean_prev)/frame_time)/ppmm_real)

    rpm = [0, 0]
    
    #i = which of the x and y axes to calculate speeds on
    for i in range(2):
        #filter unreasonably high speeds
        if speed[i] < 3000 and speed[i] >= max_speed[i]: 
            max_speed[i] = speed[i]

        ang_vel[i] = round(math.degrees(math.atan(max_speed[i]/cam_h)), 4)
        
        if ang_vel[i] > 0:
            #convert angular velocity to rpm
            rpm[i] = round(ang_vel[i]/0.1666667) #60 sec / 360˚ = 0.1666667

    if xMean < 5 and yMean < 5:
        rect_x = int((w-rect_w)/2)
        rect_y = int((h-rect_h)/2)
        wait_for_stable = 0

    # boundaries follow green dot
    if stable_limit > 0 and wait_for_stable > stable_limit and xMean >= 5 and yMean >= 5:
        rect_x = int(xMean-rect_w/2)
        rect_y = int(yMean-rect_h/2)
    
    wait_for_stable += 1

    #print("array_p = " + str(array_p) + " x: " + str(xMeanAbsolute) + " y: " + str(yMeanAbsolute))

    #paint a green follower dot
    cv2.circle(frame, (xMean, yMean), 12, (0, 255, 0), -1)
    #paint blue follower bounding rectangle
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x+rect_w, rect_y+rect_h), (255, 255, 0), 2, -1)

    #timing params printing
    # cv2.putText(frame, "zoom: " + str(round(zoom, 2)) + ", ppmm_screen: " + str(round(ppmm_screen, 2)) + ", ppmm_real: " + str(ppmm_real),
    # (10, 50), 
    # font, 
    # fontScale,
    # fontColor,
    # lineType)

    #coordinate text printing
    # cv2.putText(frame, "x: " + str(xMean) + " y: " + str(yMean) + " speed: " + str(speed) + " mm/s, max speed: " + str(max_speed),
    # (10, 600), 
    # font, 
    # fontScale,
    # fontColor,
    # lineType)

    #angular velocity text printing
    cv2.putText(frame, "angular velocity [deg/s] x: " + str(ang_vel[0]) + ", y: " + str(ang_vel[1]),
    (10, 650), 
    font, 
    fontScale,
    fontColor,
    lineType)

    #angular velocity text printing
    cv2.putText(frame, "--> rpm x: " + str(rpm[0]) + ", y: " + str(rpm[1]),
    (10, 700), 
    font, 
    fontScale,
    fontColorRPM,
    lineType)
    
    #window parameters printing
    #cv2.putText(frame, "window width: " + str(w) + " window height: " + str(h),
    #(10, 30), 
    #font, 
    #fontScale,
    #fontColor,
    #lineType)

    cv2.imshow('dst', frame)
    if ret == True:
        out.write(frame)

    #press q to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()