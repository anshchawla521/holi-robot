import cv2

# 1. Load the built-in classifiers for both front and side profiles
front_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# 2. Initialize webcam feed
cap = cv2.VideoCapture(0)

print("Starting camera... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Flip the image horizontally to catch profile views facing the other direction
    gray_flipped = cv2.flip(gray, 1)

    # 3. Multi-stage detection logic
    # Try finding frontal faces first
    faces = front_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If no front face is found, look for a profile face (facing right/left)
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If still nothing, look for profile faces facing the opposite direction using the flipped image
        if len(faces) == 0:
            flipped_faces = profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Convert coordinates back from the flipped space to the original frame space
            frame_width = frame.shape[1]
            faces = []
            for (x, y, w, h) in flipped_faces:
                actual_x = frame_width - x - w
                faces.append((actual_x, y, w, h))

    # 4. Draw bounding boxes around any detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Robust Face Detection (Front & Profile)', frame)

    # Clear window by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release system resources
cap.release()
cv2.destroyAllWindows()