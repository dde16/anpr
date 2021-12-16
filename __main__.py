import random
import numpy
import cv2
import math
import string
import colormath
import matplotlib
import os

def cv2_adjustGamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = numpy.array([
      ((i / 255.0) ** invGamma) * 255
      for i in numpy.arange(0, 256)])
   return cv2.LUT(image.astype(numpy.uint8), table.astype(numpy.uint8))

def cv2_enhance(img):
    # kernel = numpy.array([[-1,0,1],[-2,0,2],[1,0,1]])
    kernel = numpy.array([[-1,0,1],[1,0,1]])
    return cv2.filter2D(img, -1, kernel)

def anpr_normalise(image):
    out = numpy.zeros((image.shape[0], image.shape[1]))
    out  = cv2.normalize(
        image,
        out,
        0,
        8,
        norm_type=cv2.NORM_MINMAX
    )

    return out

def anpr_greyscale(image):
    return cv2_enhance(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

def anpr_threshold(image):
    _,thresh  = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # + cv2.THRESH_OTSU

    return thresh

def anpr_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_8U)

def anpr_contrast(image, clip_limit=3, grid_size=(7,7)):
    se     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # morph  = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, se)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    morph  = cv2.morphologyEx(image, cv2.MORPH_OPEN, se)

    print(morph.ndim)

    l, a, b = cv2.split(
        cv2.cvtColor(
            cv2.cvtColor(
                morph,
                cv2.COLOR_GRAY2BGR
            ),
            cv2.COLOR_BGR2LAB
        )
    )

    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))

    bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    return grey

def anpr_equalise(image):
    return cv2.equalizeHist(image)

def anpr_enhance(image, d=9, a=19, b=19):
    # grey  = cv2.GaussianBlur(grey, (5, 5), 0)
    grey  = cv2.bilateralFilter(image, d, a, b)

    return grey

def anpr_clahe(image, clip_limit=3, grid_size=(7,7)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))

    bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    return grey

def anpr_clahe_bgr(image, clip_limit=3, grid_size=(7,7)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))

    bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return bgr

def anpr_grey_2_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def anpr(path):
    SE_SHAPE = (18,3)
    ASPECT_RATIO_RANGE = (3.5, 5.5)
    MIN_WIDTH = 90
    MIN_HEIGHT = 20

    images = []

    image = cv2.imread(path)

    print(image)

    images.append({"caption": "Original", "object": image.copy()})

    image_height = image.shape[1]
    image_width  = image.shape[0]

    hsv   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg   = cv2.mean(hsv)

    del hsv

    image = anpr_normalise(image)
    images.append({"caption": "Normalisation", "object": image.copy()})

    image = anpr_greyscale(image)
    images.append({"caption": "Greyscaled", "object": anpr_grey_2_bgr(image.copy())})

    image = anpr_contrast(image)
    images.append({"caption": "Contrast", "object": image.copy()})

    image = anpr_enhance(image)
    images.append({"caption": "Enhance", "object": anpr_grey_2_bgr(image.copy())})

    # image = anpr_clahe(image)
    # images.append({"caption": "CLAHE", "object": image.copy()})

    image = anpr_equalise(image)
    images.append({"caption": "Equalised", "object": anpr_grey_2_bgr(image.copy())})

    image = anpr_threshold(image)
    images.append({"caption": "Threshold", "object": anpr_grey_2_bgr(image.copy())})

    image = anpr_laplacian(image)
    images.append({"caption": "Laplacian", "object": anpr_grey_2_bgr(image.copy())})

    se     = cv2.getStructuringElement(cv2.MORPH_RECT, SE_SHAPE)
    morph  = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)

    images.append({"caption": "Morphology", "object": morph.copy()})

    contours = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]

    image = images[0]["object"].copy()

    for contour in contours:
        contour = contour.reshape(-1,2)
        # Changes number of channels to -1 and rows to 2

        rect = cv2.minAreaRect(contour) # Minimum area for a rectangle
        box = cv2.boxPoints(rect)       # Get corners of rectangle
        box = numpy.int0(box)           # Convert to numpy array

        xs = [i[0] for i in box]        # All X values
        ys = [i[1] for i in box]        # All Y values

        x1 = min(xs)                    # Get minimum value of X values
        x2 = max(xs)                    # Get maximum value of X values
        y1 = min(ys)                    # Get minimum value of Y values
        y2 = max(ys)                    # Get maximum value of Y values

        contour_width  = x2-x1
        contour_height = y2-y1
        contour_angle  = rect[2] # Get angle of the rectangle

        # if contour_angle < -45:                 # Rotate to ~0 degrees
        #     contour_angle += 360 - contour_angle % 360    # angle += 90

        # (25 * 13)

        if (contour_width > 0 and contour_height > 0) and ((contour_width < image_width/2.0) and (contour_height < image_width/2.0)):
            if(contour_width > MIN_WIDTH and contour_height > MIN_HEIGHT):
                # Make sure the box exists and the width/height is less than half of its
                # complementary axis.

                aspect_ratio = float(contour_width/contour_height)
                # Determine aspect ratio of rectangle

                if (aspect_ratio >= ASPECT_RATIO_RANGE[0] and aspect_ratio <= ASPECT_RATIO_RANGE[1]):
                    # If the aspect ratio range is acceptable for the plate type

                    # Determine distance away from the person

                    dist = round(cv2_distanceFromCamera(110, contour_height, image_height) / 1000, 2)

                    text = str(int(contour_width)) + "x" + str(int(contour_height)) + ", " + str(round(float(contour_width/contour_height), 2)) + ", " + str(dist) + "m"
                    cv2.putText(image, text, (x1,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 160), 1, cv2.LINE_AA)

                cv2.rectangle(image, (x1, y1), (x2, y2), (200, 0, 160), 2)

    images.append({"caption": "Contours", "object": image.copy()})

    return images

    # images = [cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR), image]
    #
    # last = cv2.hconcat(images)

def cv2_distanceFromCamera(object_real_height, object_pixel_height, image_pixel_height, focal_length=4.42, sensor_height=3.342):
    return (focal_length * object_real_height * image_pixel_height) / (object_pixel_height * sensor_height)

def main():
    successes = 0
    failures  = 0
    rate      = 0.0

    # for file in os.scandir("samples"):
    # "samples/j (6).jpg"
    # images = anpr(os.path.dirname(__file__) + "/x.png")

    for f in os.listdir("samples"):
        images = anpr(os.path.dirname(__file__)+"/"+f)
        eof = False
        length = len(images)
        index = 0

        while not eof:
            image = images[index]

            cv2.putText(image["object"], str(index) + " : " + image["caption"], (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 160), 2, cv2.LINE_AA)

            cv2.imshow("1", image["object"])

            key = cv2.waitKey(0)

            if key == 10:
                index += 1

            if key == 8:
                index -= 1

            # ESC
            if key == 255:
                exit()

            eof = length == index

            cv2.destroyAllWindows()

    rate = successes / (failures + successes) * 100

    print("Success Rate : " + str(round(rate, 2)))

if __name__ == "__main__":
    main()
