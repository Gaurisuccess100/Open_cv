import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize detector and video capture
detector = HandDetector(detectionCon=0.8, maxHands=1)
video = cv2.VideoCapture(0)

# Initial position and speed of the virtual robot
x, y = 320, 240  # Start at center of frame
speed = 10
command = "Waiting for gesture"

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Flip frame horizontally (mirror effect)
    frame = cv2.flip(frame)

    # Detect hand
    hands, img = detector.findHands(frame,1)

    if hands:
        fingers = detector.fingersUp(hands[0])

        # Gesture-to-command mapping
        if fingers == [0, 1, 0, 0, 0]:
            command = "Move Forward ↑"
            y -= speed
        elif fingers == [0, 1, 1, 0, 0]:
            command = "Turn Right →"
            x += speed
        elif fingers == [0, 1, 1, 1, 0]:
            command = "Move Backward ↓"
            y += speed
        elif fingers == [0, 1, 1, 1, 1]:
            command = "Turn Left ←"
            x -= speed
        elif fingers == [1, 1, 1, 1, 1]:
            command = "Stop ✋"
        else:
            command = "Unknown Gesture"

    # Keep the robot within window bounds
    x = max(20, min(x, 620))
    y = max(20, min(y, 460))

    # Draw the virtual robot (circle)
    cv2.circle(img, (x, y), 30, (255, 0, 0), -1)

    # Draw command text
    cv2.putText(img, f"Command: {command}", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Virtual Robot Controller", img)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


