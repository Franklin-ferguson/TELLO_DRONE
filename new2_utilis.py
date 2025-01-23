from djitellopy import Tello
import cv2
import numpy as np

def initializeTello():
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0
    print("Battery Level:", myDrone.get_battery())
    myDrone.streamoff()
    myDrone.streamon()
    return myDrone

def telloGetFrame(myDrone, w=360, h=240):
    myFrame = myDrone.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (w, h))
    return img

def findFace(img):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 4)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = int(x + w // 2)
        cy = int(y + h // 2)
        area = w * h
        myFaceListArea.append(area)
        myFaceListC.append([cx, cy])

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

def trackFace(myDrone, info, w, pid, pError):
    target_area = 4000  # Desired face area to maintain distance
    tolerance = 500     # Allowed deviation for stability
    center_margin = 30  # Deadzone for horizontal tracking

    # Horizontal error (for yaw control)
    x_error = float(info[0][0]) - w // 2

    # Distance error (based on face area)
    area_error = target_area - info[1]

    # Calculate yaw speed using PID control
    if abs(x_error) > center_margin:  # Check if error exceeds deadzone
        yaw_speed = pid[0] * x_error + pid[1] * (x_error - pError)
        yaw_speed = int(np.clip(yaw_speed, -100, 100))
    else:
        yaw_speed = 0

    # Adjust forward/backward velocity based on face area
    if abs(area_error) > tolerance:  # Move only if outside tolerance
        if area_error > 0:  # Face is too far, move forward
            fb_speed = 20
        else:  # Face is too close, move backward
            fb_speed = -20
    else:
        fb_speed = 0

    print(f"Yaw Speed: {yaw_speed}, Forward/Back Speed: {fb_speed}, Area Error: {area_error}")

    # Send commands to the drone
    if info[0][0] != 0:  # If a face is detected
        myDrone.yaw_velocity = yaw_speed
        myDrone.for_back_velocity = fb_speed
    else:  # No face detected, stop all movement
        myDrone.yaw_velocity = 0
        myDrone.for_back_velocity = 0
        myDrone.left_right_velocity = 0
        myDrone.up_down_velocity = 0

    if myDrone.send_rc_control:
        myDrone.send_rc_control(
            myDrone.left_right_velocity,
            myDrone.for_back_velocity,
            myDrone.up_down_velocity,
            myDrone.yaw_velocity,
        )

    return x_error

# Function to load the saved face area (for future reference)
def load_face_area():
    try:
        with open('face_area.txt', 'r') as f:
            return int(f.read())
    except FileNotFoundError:
        print("No saved face area found, using default values.")
        return 0  # Default value if no saved area is found
