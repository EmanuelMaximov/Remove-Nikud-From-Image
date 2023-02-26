import sys
import cv2


def main(argv):
    if len(argv) < 1:
        print('\nNot enough parameters!')
        print('Usage:\npython main.py <path_to_image>')
        return -1

    # Load the image
    img = cv2.imread(argv[0], cv2.IMREAD_COLOR)

    # Check if image is loaded fine
    if img is None:
        print('Error opening image: ' + argv[0])
        return -1

    # Image Background Sample
    b, g, r = (img[0, 0])
    img_background = (int(b), int(g), int(r))

    # Convert to Grayscale for further thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Show source image
    cv2.imshow("Source Image", img)

    # Threshold - Convert to Mask (Binary Image)
    # Threshold accordingly to image size
    if img.shape[1] < 600:
        thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)[1]
    else:
        thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    contour_indexes = []
    heights = set()
    widths = set()
    for c in contours:
        _, _, w, h = cv2.boundingRect(c)
        heights.add(h)
        widths.add(w)

    # Remove noise contours
    heights = list(filter(lambda a: a > 2, heights))
    widths = list(filter(lambda a: a > 2, widths))

    # Find the Maximum Height and Width of all contours
    heights.sort()
    widths.sort()
    max_height = heights[-1]
    max_width = widths[-1]
    # The ratio between biggest letter (ץ,ף,ק,ן) and smallest letter (י)
    width_ratio = 0.5
    height_ratio = 0.35
    for index, cnt in enumerate(contours):
        # Extract Contour width and height
        _, _, w, h = cv2.boundingRect(cnt)
        # Mark the demarcation for removing
        if not (w > max_width * width_ratio or h > max_height * height_ratio):
            contour_indexes.append(index)

    for i in contour_indexes:
        cv2.drawContours(image=img, contours=contours, contourIdx=i, color=img_background, thickness=cv2.FILLED,
                         lineType=cv2.LINE_AA)
        cv2.drawContours(image=img, contours=contours, contourIdx=i, color=img_background, thickness=2,
                         lineType=cv2.LINE_AA)

    cv2.imshow('Without demarcation', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
