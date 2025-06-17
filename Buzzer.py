import cv2
import numpy as np
import winsound

# Initialize the USB webcam (0 is typically the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Could not read frame")
    cap.release()
    exit()

# Convert the first frame to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

try:
    while True:
        # Read the next frame
        ret, frame2 = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Convert the new frame to grayscale and blur it
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

        # Compute the absolute difference between frames
        delta = cv2.absdiff(gray1, gray2)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ignore small movements
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                motion_detected = True

        # Display alert and play sound if motion is detected
        if motion_detected:
            cv2.putText(frame2, "Motion Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            winsound.Beep(1000, 100)  # 1000 Hz, 100 ms beep

        # Display the frame and threshold
        cv2.imshow("Motion Detection", frame2)
        cv2.imshow("Threshold", thresh)

        # Update the previous frame
        gray1 = gray2

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program terminated by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()