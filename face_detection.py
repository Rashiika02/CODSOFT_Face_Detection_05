import cv2
import face_recognition

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known face encodings and names
# Example: Load your own face encodings here
known_face_encodings = []
known_face_names = []

def load_known_faces():
    # Load a sample known face image and encode it
    image = face_recognition.load_image_file('known_face.jpg')  # Replace with your image file
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append("Person Name")  # Replace with the person's name

load_known_faces()

def detect_and_recognize_faces(image):
    # Convert image to grayscale for Haar cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        face_encoding = face_recognition.face_encodings(face_image)
        
        if face_encoding:
            face_encoding = face_encoding[0]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Check if face matches known faces
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw rectangle and label on the image
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image

# Load an image to process
image = cv2.imread('input_image.jpg')  # Replace with your input image file

# Detect and recognize faces in the image
output_image = detect_and_recognize_faces(image)

# Show the result
cv2.imshow('Face Detection and Recognition', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('output_image.jpg', output_image)  # Save the output image with faces labeled
