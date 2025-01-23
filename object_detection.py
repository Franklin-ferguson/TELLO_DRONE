# <<<<<<<<< T E L L O   D R O N E  >>>>>>>>>>>>
# <<<<<<<< OBJECT DETECTION >>>>>>>>>>>

# <<<<<<<< COMMANDS >>>>>>>
# Takeoff (T key)
# Land (L key)
# Move Forward (W key)
# Move Backward (S key)
# Move Left (A key)
# Move Right (D key)
# Move Up (Up Arrow key)
# Move Down (Down Arrow key)
# Rotate left by 30 degrees (Left Arrow key)
# Rotate right by 30 degrees (Right Arrow Key)
# Emergency Stop (P key)

from djitellopy import tello
import cv2
import cvzone
import keyboard

thres = 0.55
nmsThres = 0.2

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamoff()
me.streamon()

BATTERY_LEVEL = me.get_battery()
speed = 50


def move_drone():
    # Takeoff (T key)
    if keyboard.is_pressed('t'):  # Takeoff
        if not me.is_flying:
            me.takeoff()
            print("Taking off...")

    # Land (L key)
    elif keyboard.is_pressed('l'):  # Land
        if me.is_flying:
            me.land()
            print("Landing...")

    # Move Forward (W key)
    elif keyboard.is_pressed('w'):  # Forward
        if me.is_flying:
            me.move_forward(speed)
            print("Moving Forward")

    # Move Backward (S key)
    elif keyboard.is_pressed('s'):  # Backward
        if me.is_flying:
            me.move_back(speed)
            print("Moving Backward")

    # Move Left (A key)
    elif keyboard.is_pressed('a'):  # Left
        if me.is_flying:
            me.move_left(speed)
            print("Moving Left")

    # Move Right (D key)
    elif keyboard.is_pressed('d'):  # Right
        if me.is_flying:
            me.move_right(speed)
            print("Moving Right")

    # Move Up (Up Arrow key)
    elif keyboard.is_pressed('up'):  # Up
        if me.is_flying:
            me.move_up(speed)
            print("Moving Up")

    # Move Down (Down Arrow key)
    elif keyboard.is_pressed('down'):  # Down
        if me.is_flying:
            me.move_down(speed)
            print("Moving Down")

    # Rotate left by 30 degrees
    elif keyboard.is_pressed('left'):
        if me.is_flying:
            # Rotate counterclockwise (yaw left)
            me.rotate_counter_clockwise(30)  # Rotate left by 30 degrees
            print("Rotate Left")

    # Rotate right by 30 degrees
    elif keyboard.is_pressed('right'):
        # Rotate clockwise (yaw right)
        if me.is_flying:
            me.rotate_clockwise(30)
            print("Rotate Left")

        # Emergency Stop (P key)
    elif keyboard.is_pressed('p'):  # Emergency stop
        me.emergency()
        print("Emergency stop!")


while True:

    img = me.get_frame_read().frame
    img = cv2.resize(img, (640, 360))
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cvzone.cornerRect(img, box)
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)
    except:
        pass

    print(f"<<<BATTERY>>> : {BATTERY_LEVEL}")
    move_drone()

    cv2.imshow("Image", img)
    cv2.waitKey(1)
