import cv2
from cvzone.HandTrackingModule import HandDetector
from directkeys import press_key, release_key, space_pressed
import time

# Initialize detector
detector = HandDetector(detectionCon=0.8, maxHands=1)
space_key_pressed = space_pressed
current_key_pressed = set()

# Start camera
video = cv2.VideoCapture(0)
video.set(3, 640)
video.set(4, 480)

# Wait for camera
time.sleep(2.0)

if not video.isOpened():
    print(" Error: Could not open webcam")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        print("  Failed to grab frame")
        break

    frame = cv2.flip(frame, 1) 
    hands, img = detector.findHands(frame)

    cv2.rectangle(img, (0, 480), (300, 425), (50, 50, 255), -2)
    cv2.rectangle(img, (640, 480), (400, 425), (50, 50, 255), -2)

    key_pressed = False

    if hands:
        fingerUp = detector.fingersUp(hands[0])
        print(f"Fingers up: {fingerUp}")

        if fingerUp == [0, 0, 0, 0, 0]:
            cv2.putText(img, 'Finger Count: 0', (20, 460),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, 'Jumping', (440, 460),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if space_key_pressed not in current_key_pressed:
                press_key(space_key_pressed)
                current_key_pressed.add(space_key_pressed)

            key_pressed = True

        else:
            cv2.putText(img, f'Finger Count: {fingerUp.count(1)}', (20, 460),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, 'Not Jumping', (420, 460),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Release key if not pressed
    if not key_pressed and space_key_pressed in current_key_pressed:
        release_key(space_key_pressed)
        current_key_pressed.remove(space_key_pressed)

    cv2.imshow("Hand Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
