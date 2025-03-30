import cv2                                # OpenCV for video processing
import mediapipe as mp                    # MediaPipe for hand tracking
from math import hypot                    # For calculating the distance between two points
import screen_brightness_control as sbc   # To control screen brightness
import numpy as np                        # For numerical operations

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand Tracking
mpHands = mp.solutions.hands  # Load MediaPipe Hands module
hands = mpHands.Hands()  # Create a hand tracking object
mpDraw = mp.solutions.drawing_utils  # Utility to draw hand landmarks

while True:
    # Capture frame from webcam
    success, img = cap.read()
    if not success:
        break  # Exit loop if the frame is not captured
    
    img = cv2.flip(img, 1)  # Flip horizontally for natural hand movement
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
    results = hands.process(imgRGB)  # Process the frame for hand detection
    
    lmList = []  # List to store hand landmark positions
    
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape  # Get frame dimensions
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coords to pixel values
                lmList.append([id, cx, cy])  # Store landmark ID and position
                
            # Draw hand landmarks on the image
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)
    
    # If landmarks are detected, calculate brightness based on thumb-index finger distance
    if lmList:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip coordinates
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip coordinates
        
        # Draw circles at thumb and index finger tips
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        
        # Draw line connecting thumb and index finger
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        
        # Calculate the Euclidean distance between the two fingers
        length = hypot(x2 - x1, y2 - y1)
        
        # Map the distance to brightness range (15-220 pixels -> 0-100% brightness)
        bright = int(round(np.interp(length, [15, 220], [0, 100])))
        
        # Set the screen brightness
        sbc.set_brightness(bright)
        
        # Draw a rectangle to display brightness level
        cv2.rectangle(img, (30, 30), (350, 100), (0, 0, 0), cv2.FILLED)  # Black box
        cv2.putText(img, f"Brightness: {bright}%", (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 3)  # Green text
    
    # Display the processed frame
    cv2.imshow('Hand Brightness Control', img)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close OpenCV window
cap.release()
cv2.destroyAllWindows()
