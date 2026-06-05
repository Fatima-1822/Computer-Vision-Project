import cv2

# Open the default webcam
cap = cv2.VideoCapture(0)

while True:
    # Read one frame from the webcam
    ret, frame = cap.read()

    # If camera cannot be read, stop the program
    if not ret:
        print("Camera not found")
        break

    # Show the webcam frame in a window
    cv2.imshow("Camera Test", frame)

    # Press q to quit
    if cv2.waitKey(1) == ord("q"):
        break

# Release camera and close window
cap.release()
cv2.destroyAllWindows()
