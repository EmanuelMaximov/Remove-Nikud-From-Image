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
    # thresh2 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # draw contours on the original image
    image_copy = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY_INV)[1]

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

    area_ratio = 1
    max_height = heights[-1]
    max_width = widths[-1]

    ratio_h = heights[4] / heights[-1]
    ratio_w = widths[4] / widths[-1]
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
    #
    # # sort the list based on the contour size.
    # # this changes the order of the elements in the list
    # contour_sizes.sort(key=lambda x: x[1])
    # op1 = np.array([[1, 1],
    #                 [1, 1]], dtype=thresh.dtype)
    # op2 = np.array([[0, 1],
    #                 [1, 0]], dtype=thresh.dtype)
    #
    # op3 = np.array([[0, 0, 1], [0, 1, 1],
    #                 [1, 1, 1]], dtype=thresh.dtype)
    #
    # se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    # se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    # se4 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # se3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # se6 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # # mask1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, se1)
    # # mask1 = cv2.erode(thresh, se2, 2)
    # mask1 = cv2.erode(thresh, se3, iterations=1)
    # mask1 = cv2.dilate(mask1, se3, iterations=1)
    # mask1 = cv2.erode(mask1, se6, iterations=1)
    # mask1 = cv2.dilate(mask1, se3, iterations=1)
    # mask1 = cv2.erode(mask1, se3, iterations=1)
    # # mask1 = cv2.dilate(mask1, se3, 1)
    # mask2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, se2)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se1)

    # cv2.imshow('mask1', mask1)
    # # cv2.imshow('mask2', mask2)
    # # cv2.imshow('mask3', cv2.bitwise_or(mask1, mask2))
    # # cv2.imshow('mask2', mask2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # mask = cv2.dilate(mask, op6, iterations=1)

    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, rect2)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse)
    # mask = cv2.erode(mask, rect3, iterations=1)
    # mask = cv2.dilate(mask, rect, iterations=2)
    # mask = cv2.erode(mask, ellipse, iterations=2)

    # mask = cv2.erode(mask, rect, iterations=2)
    # mask = cv2.dilate(mask, rect, iterations=1)

    # mask = cv2.erode(mask, ellipse, iterations=1)
    # mask = cv2.dilate(mask, ellipse, iterations=2)
    # mask = cv2.erode(mask, element5, iterations=3)
    # mask = cv2.dilate(mask, element5, iterations=3)
    # mask = cv2.erode(mask, element5, iterations=2)
    # mask = cv2.dilate(mask, element5, iterations=4)
    # mask = cv2.dilate(mask, element6, iterations=1)
    # mask = cv2.dilate(mask, element3, iterations=1)
    # mask = cv2.erode(mask, element4, iterations=1)
    # mask = cv2.dilate(mask, element4, iterations=1)

    # mask = cv2.absdiff(thresh, mask)
    # mask = cv2.erode(mask, element2, iterations=1)
    # mask = cv2.dilate(mask, element2, iterations=1)
    # mask = cv2.erode(thresh, element3, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)

    # mask = cv2.dilate(mask, element2, iterations=3)
    # see the results
    # cv2.imshow('mask1', mask)
    # cv2.imshow('mask2', mask2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # For help:
    # element = cv.getStructuringElement(cv.MORPH_RECT, (8, 1))
    # element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # element1 = np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=bw.dtype)
    # element = np.array([[1], [1],[1], [1], [1], [1]], dtype=bw.dtype)
    # edges = cv2.Canny(image=src, threshold1=100, threshold2=200)
    # mask = cv2.erode(bw, element1, iterations=1)
    # mask = cv2.dilate(mask, element, iterations=1)
    # mask = cv.morphologyEx(bw, cv.MORPH_OPEN, element)
    # gb = cv.GaussianBlur(src, (3, 3), 0)
    # sobel = cv.Sobel(gray, ddepth=-1, dx=1, dy=1, ksize=3, scale=1, delta=0)
    # btand = cv.bitwise_and(mask, bw)

    return 0


if __name__ == "__main__":
    main(["test_num_1.png"])
