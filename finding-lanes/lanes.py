import cv2
import matplotlib.pyplot as plt
import numpy as np


def canny(image):
    """ Convert an RGB image to a Canny image.
    Inputs:
        image: [numpy.ndarray] input RGB image.
    Outputs:
        canny_img: [numpy.ndarray] output processed image.
    """
    # Convert to grayscale image and display
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('Grayscale', gray)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Grayscale')

    # Smooth and display the grayscale image with a Gaussian Blur
    # Using a 5x5 convolutional matrix
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow('Smoothened grayscale', blur)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Smoothened grayscale')

    # Detect the edge with a canny function
    # Setting the lower and upper thresholds
    canny_img = cv2.Canny(blur, 50, 150)
    # cv2.imshow('Canny image', canny_img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Canny image')

    return canny_img


def region_of_interest(image):
    """ Reshape the field of view in a triangular shape around the region of interest.
    Inputs:
        image: [numpy.ndarray] input image.
    Outputs:
        mask: [numpy.ndarray] output processed image.
    """
    # Get the height of the image
    height = image.shape[0]
    # Set the 3 points of the triangle
    triangle = np.array([[(200, height), (1100, height), (550, 250)]])
    # Create an array of zeros the size of the image
    mask = np.zeros_like(image)
    # Fill the black mask with the white triangle
    cv2.fillPoly(mask, triangle, 255)
    # Apply the mask on the image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    """ Display lines on top of the image.
    Inputs:
        image: [numpy.ndarray] input image.
        lines: [numpy.ndarray] array of lines coordinates.
    Outputs:
        line_image: [numpy.ndarray] image with the lines.
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # Reshape the line array
            x1, y1, x2, y2 = line.reshape(4)
            # Draw the line
            cv2.line(line_image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=10)
    return line_image


def average_slope_intercept(image, lines):
    """ Average the lines of the image and return 2 lines.
    Inputs:
        image: [numpy.ndarray] input image.
        lines: [numpy.ndarray] array of lines coordinates.
    Outputs:
        [numpy.ndarray] averaged left and right lines.
    """
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if len(left_fit) and len(right_fit):
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])


def make_coordinates(image, line_parameters):
    """ Calculates coordinates out of line slope and intercept parameters.
    Inputs:
        image: [numpy.ndarray] input image.
        line_parameters: [numpy.ndarray] array of line parameters.
    Outputs:
        [numpy.ndarray] coordinates of the line.
    """
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

# Load the image and Numpy copy
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

# Display the original image
cv2.imshow('Original', lane_image)
cv2.waitKey(0)
cv2.destroyWindow('Original')

# Create a canny version of the image
canny_img = canny(lane_image)
# Define a region of interest in the image
cropped_img = region_of_interest(canny_img)
# Detect the lines in the image using a Hough transform
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40,
                        maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines)
line_img = display_lines(lane_image, averaged_lines)

# Apply the lines on the image
combo_image = cv2.addWeighted(lane_image, 0.8, line_img, 1, 1)

# Display the image
cv2.imshow('Result', combo_image)
cv2.waitKey(0)
cv2.destroyWindow('Result')

cap = cv2.VideoCapture('test2.mp4')
while cap.isOpened():
    _, frame = cap.read()
    # Create a canny version of the image
    canny_img = canny(frame)
    # Define a region of interest in the image
    cropped_img = region_of_interest(canny_img)
    # Detect the lines in the image using a Hough transform
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40,
                            maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_img = display_lines(frame, averaged_lines)
    # Apply the lines on the image
    combo_image = cv2.addWeighted(frame, 0.8, line_img, 1, 1)

    # Display the image
    cv2.imshow('Result', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()