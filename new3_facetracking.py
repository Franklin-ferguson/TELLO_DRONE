# <<<<<<<<< T E L L O   D R O N E  >>>>>>>>>>>>
# <<<<<<<< FACE TRACKING >>>>>>>>>>>

# <<<<<<<< COMMANDS >>>>>>>
# Takeoff (T key)
# Land (L key)



from new2_utilis import *
import cv2
import keyboard  # Import the keyboard library

w, h = 360, 240
pid = [0.4, 0.4, 0]  # PID values
pError = 0
startCounter = 1  # Set to 0 for flight, 1 for testing without flight

# Initialize Tello drone
myDrone = initializeTello()

# Function to save the face area to a file


def save_face_area(area):
    try:
        with open('face_area.txt', 'w') as f:
            f.write(str(area))
        print(f"Saved face area: {area}")
    except Exception as e:
        print(f"Error saving face area: {e}")

# Function to trigger manual calibration with the 'C' key
def manual_calibration_trigger(img, info):
    # Detect if the 'C' key is pressed using the keyboard library
    if keyboard.is_pressed('c'):  # Check if 'C' is pressed
        print("C key pressed - Saving face area...")
        save_face_area(info[1])

# Function to handle landing when 'A' key is pressed


def manual_land_trigger():
    if keyboard.is_pressed('l'):  # Check if 'A' is pressed
        print("A key pressed - Landing the drone...")
        myDrone.land()  # Land the drone
        return True  # Stop the loop
    return False  # Continue the loop


# Main loop
while True:
    if keyboard.is_pressed('t'):  # Check if 'A' is pressed
        print("A key pressed - Landing the drone...")
        myDrone.takeoff()
        startCounter = 1

    # Step 1: Capture the frame
    img = telloGetFrame(myDrone, w, h)

    # Step 2: Detect face and get position/area info
    img, info = findFace(img)
    print(f"Face Info: Center = {info[0]}, Area = {info[1]}")

    # Step 3: Trigger manual calibration by pressing 'C'
    manual_calibration_trigger(img, info)

    # Step 4: Track the face
    pError = trackFace(myDrone, info, w, pid, pError)

    # Step 5: Display the video feed with face area
    cv2.putText(img, f"Face Area: {info[1]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Image', img)

    # Step 6: Check if 'A' is pressed for landing
    if manual_land_trigger():
        break  # Exit the loop after landing

    # Allow OpenCV to update the window and check for key presses
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

cv2.destroyAllWindows()
