import cv2
import sys

# cascPath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(
    0)  # This line sets the video source to the default webcam, which OpenCV can easily capture.

while True:
    # capture frame - by - frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        # flags=cv2.CV_HAAR_SCALE_IMAGE
    )

    # Draw a recangular around faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):  # We wait for the ‘q’ key to be pressed. If it is, we exit the script.

        break

# when everything is done release the capture
# Here, we are just cleaning up.
video_capture.release()
cv2.destroyAllWindows()
