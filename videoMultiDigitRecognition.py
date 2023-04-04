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

    # Threshold the grayscale image to extract the black digits
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find the contours of the black pixels
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and extract the bounding boxes of the digits
    digits = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w >= 5 and h >= 25:
            digit = gray[y:y + h, x:x + w]
            digit = cv2.resize(digit, (28, 28))
            digits.append(digit)

    # If any digits were found, use the model to predict the digits in the captured frame
    if digits:
        # Convert the digits to a PyTorch tensor
        tensor = torch.from_numpy(np.array(digits)).float()

        # Use the model to predict the digits
        output = model(tensor)

        # Get the predicted digits
        predicted_digits = output.argmax(dim=1)

        # Loop through the predicted digits and add them to the frame
        i = 0
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w >= 5 and h >= 25:
                digit = predicted_digits[i].item()
                cv2.putText(frame, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                i += 1

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
