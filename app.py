import cv2
import numpy as np

def detect_and_crop_card(image_path):
    # Read the image
    image = cv2.imread(image_path)
    original = image.copy()
    
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blur, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Loop over the contours
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # If our approximated contour has four points, we can assume it's the card
        if len(approx) == 4:
            card_contour = approx
            break
    
    # Draw the contour of the card on the image
    cv2.drawContours(image, [card_contour], 0, (0, 255, 0), 2)
    
    # Get the bounding rectangle of the card
    x, y, w, h = cv2.boundingRect(card_contour)
    
    # Crop the card from the original image
    card = original[y:y+h, x:x+w]
    
    return image, card

# Usage
image_path = '/Users/shrawanthakur/Downloads/vc.jpg'
marked_image, cropped_card = detect_and_crop_card(image_path)

# Display results
cv2.imshow('Marked Image', marked_image)
cv2.imshow('Cropped Card', cropped_card)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save results
cv2.imwrite('marked_image.jpg', marked_image)
cv2.imwrite('cropped_card.jpg', cropped_card)