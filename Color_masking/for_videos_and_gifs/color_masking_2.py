import cv2 as cv
import numpy as np

def color_masking(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    lower_brown = np.array([6, 50, 30])
    upper_brown = np.array([20, 255, 110])

    bin_mask = cv.inRange(hsv, lower_brown, upper_brown)
    # cv.imshow("Before closing", bin_mask)
    bin_mask = cv.GaussianBlur(bin_mask, (1, 1), 0)

    kernel1 = np.ones((27, 27), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)

    # Apply erosion followed by dilation (closing)
    closing = cv.morphologyEx(bin_mask, cv.MORPH_CLOSE, kernel2)

    # Apply dilation
    # mask3 = cv.dilate(closing, kernel2, iterations=1)
    cv.imshow("Closing", closing)
    result = cv.bitwise_and(image, image, mask=closing)

    return result

# video_path = "./videos_and_gifs/Find LINE FRIENDS at STARBUCKS _ BROWN & FRIENDS.mp4"  
# video_path = "./videos_and_gifs/b757c6433fb9a47.gif"  
video_path = "./videos_and_gifs/5e4f44ceabadd4e7dfe2c9cd676caffe.gif"  
cap = cv.VideoCapture(video_path)

# Get the frames per second (fps) of the input video
fps = cap.get(cv.CAP_PROP_FPS) if cap.get(cv.CAP_PROP_FPS) else 30  

original_frame_width = int(cap.get(3)) if cap.get(3) else 640
original_frame_height = int(cap.get(4)) if cap.get(4) else 480

frame_width = int(original_frame_width * 0.5) 
frame_height = int(original_frame_height * 0.5)

delay = int(1000 / fps) 

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resizing the frame


    # frame = cv.resize(frame, (frame_width, frame_height))

    result_frame = color_masking(frame)

    cv.imshow("Frame", frame)
    cv.imshow("Result Frame", result_frame)

    if cv.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()

cv.destroyAllWindows()
