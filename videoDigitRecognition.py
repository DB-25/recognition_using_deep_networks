import cv2
import numpy as np
import torch

# Load the PyTorch model
model = torch.load("path/to/model.pt")

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to extract the white pixels
    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    # Find the contours of the white pixels
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (which should be the white sheet of paper)
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # If a contour was found, extract the white sheet of paper from the video stream
    if max_contour is not None:
        # Create a mask of the white sheet of paper
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [max_contour], 0, 255, -1)

        # Apply the mask to the original color image
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert the masked frame to grayscale
        masked_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        # Resize the masked grayscale frame to the same size as the images used to train the PyTorch model
        resized = cv2.resize(masked_gray, (28, 28))

        # Convert the resized frame to a PyTorch tensor
        tensor = torch.from_numpy(np.array([resized])).float()

        # Use the model to predict the digit in the captured frame
        output = model(tensor)

        # Get the predicted digit
        digit = output.argmax().item()

        # Add the predicted digit to the frame
        cv2.putText(frame, str(digit), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Video", frame)

    # Wait for the user to press a key
    key = cv2.waitKey(1)

    # If the user presses the "q" key, exit the loop
    if key == ord("q"):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
