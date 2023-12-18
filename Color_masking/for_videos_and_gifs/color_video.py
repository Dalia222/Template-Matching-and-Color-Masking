import cv2 as cv
import numpy as np

video_path = "./videos_and_gifs/ALL ABOUT BROWN Ep.01 BROWN is so cute _ BROWN & FRIENDS.mp4"

def color_masking(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # lower_brown = np.array([6, 50, 30])
    # upper_brown = np.array([20, 255, 110])
    lower_brown = np.array([4, 70, 30])
    upper_brown = np.array([15, 200, 110])
        # Create a binary mask
    bin_mask = cv.inRange(hsv, lower_brown, upper_brown)
    bin_mask = cv.GaussianBlur(bin_mask, (1, 1), 0)

    kernel1 = np.ones((9, 9), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)

    # Apply erosion followed by dilation (closing)
    closing = cv.morphologyEx(bin_mask, cv.MORPH_CLOSE, kernel1)

    # Apply dilation
    # mask3 = cv.dilate(closing, kernel2, iterations=1)

    result = cv.bitwise_and(image, image, mask=closing)

    return result, closing

cap = cv.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result_frame, detected_objects_mask = color_masking(frame)

    contours, _ = cv.findContours(detected_objects_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bounding_rect = None
    for contour in contours:
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        x, y, w, h = cv.boundingRect(contour)
        if bounding_rect is None:
            bounding_rect = cv.boundingRect(box)
        else:
            bounding_rect = (
                min(bounding_rect[0], min(box[:, 0])),
                min(bounding_rect[1], min(box[:, 1])),
                max(bounding_rect[0] + bounding_rect[2], max(box[:, 0])) - min(bounding_rect[0], min(box[:, 0])),
                max(bounding_rect[1] + bounding_rect[3], max(box[:, 1])) - min(bounding_rect[1], min(box[:, 1]))
            )
    if bounding_rect is not None:
        cv.rectangle(result_frame, (bounding_rect[0], bounding_rect[1]),
                     (bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]), (0, 255, 0), 2)

    cv.imshow('Video with Brown Object Detection', result_frame)

    if cv.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break
        #cv.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv.drawContours(result_frame, [box], 0, (0, 255, 0), 2)
    # cv.imshow('Video with Brown Object Detection', result_frame)

    # if cv.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
    #     break

cap.release()
cv.destroyAllWindows()

