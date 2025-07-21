import cv2 
import mediapipe as mp
import pyautogui
import time

# Initialize camera and face mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Track last click time to avoid repeated clicks
click_time = 0

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip horizontally for natural movement
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame.shape

    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Move mouse using iris center (landmark 475)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            if id == 1:  # landmark 475
                screen_x = int(landmark.x * screen_w)
                screen_y = int(landmark.y * screen_h)
                pyautogui.moveTo(screen_x, screen_y)

        # Blink detection using left eye landmarks: 145 (bottom), 159 (top)
        eye_top = landmarks[159]
        eye_bottom = landmarks[145]
        x1 = int(eye_top.x * frame_w)
        y1 = int(eye_top.y * frame_h)
        x2 = int(eye_bottom.x * frame_w)
        y2 = int(eye_bottom.y * frame_h)

        # Draw circles on eye points
        cv2.circle(frame, (x1, y1), 3, (255, 0, 255), -1)
        cv2.circle(frame, (x2, y2), 3, (255, 0, 255), -1)

        # Calculate eye distance
        eye_distance = abs(eye_top.y - eye_bottom.y)
        cv2.putText(frame, f'Distance: {eye_distance:.4f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Click if eye distance is below threshold (blink)
        if eye_distance < 0.01 and (time.time() - click_time) > 1:
            print("Click triggered (blink)")
            pyautogui.click()
            click_time = time.time()

    # Show frame
    cv2.imshow("Eye Control Mouse", frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()


