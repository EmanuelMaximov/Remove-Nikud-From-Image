import numpy as np
import sys
import cv2


def main(argv):
    if len(argv) < 1:
        print('Not enough parameters')
        print('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1

    # Load the image
    img = cv2.imread(argv[0], cv2.IMREAD_COLOR)
    b, g, r = (img[0, 0])
    img_background = (int(b), int(g), int(r))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check if image is loaded fine
    if img is None:
        print('Error opening image: ' + argv[0])
        return -1

    # Show source image
    cv2.imshow("Source Image", img)

    # Convert to Mask (Binary Image)
    thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    contour_indexes = []

    heights = set()
    widths = set()
    cnt_areas = set()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        heights.add(h)
        widths.add(w)
        cnt_areas.add(int(cv2.contourArea(c)))
        # cv2.putText(image_copy, str(w), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 1)
    heights = list(filter(lambda a: a > 2, heights))
    widths = list(filter(lambda a: a > 2, widths))
    cnt_areas = list(filter(lambda a: a > 2, cnt_areas))
    heights.sort()
    widths.sort()
    cnt_areas.sort()

    max_height = heights[-1]
    max_width = widths[-1]

    for index, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if not (w > max_width * 0.5 or h > max_height * 0.35):
            contour_indexes.append(index)

    for i in contour_indexes:
        cv2.drawContours(image=img, contours=contours, contourIdx=i, color=img_background, thickness=cv2.FILLED,
                         lineType=cv2.LINE_AA)
        cv2.drawContours(image=img, contours=contours, contourIdx=i, color=img_background, thickness=2,
                         lineType=cv2.LINE_AA)

    cv2.imshow('Without demarcation', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # element1 = np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=bw.dtype)
    # edges = cv2.Canny(image=src, threshold1=100, threshold2=200)
    # mask = cv2.erode(bw, element1, iterations=1)
    # mask = cv2.dilate(mask, element, iterations=1)
    # mask = cv.morphologyEx(bw, cv.MORPH_OPEN, element)
    # btand = cv.bitwise_and(mask, bw)

    return 0


if __name__ == "__main__":
    main(["test_num_1.png"])
