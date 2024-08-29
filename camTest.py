import cv2

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Camera not accessible")
else:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) % 256 == 27:  # ESC pressed
            break

    cam.release()
    cv2.destroyAllWindows()
