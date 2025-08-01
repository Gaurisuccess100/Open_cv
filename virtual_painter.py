import cv2
import numpy as np
import mediapipe as mp

# Load toolbar header image
header = cv2.imread("2.png - Copy (2).jpg")
header = cv2.resize(header, (1280, 125))

# Brush settings
drawColor = (255, 0, 255)  # Default pink
brushThickness = 15
eraserThickness = 50

# Variables for drawing
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Setup MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Process hand
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        for id, lm in enumerate(handLms.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append((cx, cy))
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if lmList:
        x1, y1 = lmList[8]   # Index finger
        x2, y2 = lmList[12]  # Middle finger

        fingers = []
        fingers.append(lmList[4][0] > lmList[3][0])  # Thumb
        for tip in [8, 12, 16, 20]:
            fingers.append(lmList[tip][1] < lmList[tip - 2][1])

        # Selection mode: two fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            if y1 < 125:
                if 250 < x1 < 350:
                    drawColor = (255, 0, 255)
                elif 400 < x1 < 500:
                    drawColor = (255, 0, 0)
                elif 550 < x1 < 650:
                    drawColor = (0, 255, 255)
                elif 700 < x1 < 800:
                    drawColor = (50, 50, 50)
                elif 850 < x1 < 950:
                    drawColor = (0, 0, 0)  # Eraser

        # Drawing mode: only index finger up
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    # Merge canvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Overlay header
    img[0:125, 0:1280] = header

    cv2.imshow("Painter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




