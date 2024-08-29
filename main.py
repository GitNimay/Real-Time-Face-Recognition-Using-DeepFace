import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace

# Path to store face data
BASE_DIR = "face_database"

# Create the base directory if it doesn't exist
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

# Load OpenCV's pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to capture and store face data
def capture_face_data(name, max_images=5):
    user_dir = os.path.join(BASE_DIR, name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Face Data")

    count = 0
    while count < max_images:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            face_path = os.path.join(user_dir, f"{name}_{count}.jpg")
            cv2.imwrite(face_path, face_image)
            print(f"Saved {face_path}")
            count += 1
            if count >= max_images:
                break

        cv2.imshow("Capture Face Data", frame)
        if cv2.waitKey(1) % 256 == 27:  # ESC pressed to exit early
            break

    cam.release()
    cv2.destroyAllWindows()

# Function to recognize faces in real-time
def recognize_faces():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Real-Time Face Recognition")

    # Load all faces in the dataset
    face_db = {}
    for user_dir in os.listdir(BASE_DIR):
        user_path = os.path.join(BASE_DIR, user_dir)
        if os.path.isdir(user_path):
            for filename in os.listdir(user_path):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(user_path, filename)
                    face_db[img_path] = user_dir

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            face_image_resized = cv2.resize(face_image, (224, 224))  # Resize for consistency

            # Find the closest match in the database
            best_match = None
            min_distance = float('inf')
            for img_path, name in face_db.items():
                try:
                    result = DeepFace.verify(face_image_resized, img_path, enforce_detection=False)
                    distance = result["distance"]
                    if distance < min_distance:
                        min_distance = distance
                        best_match = name
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

            # Draw bounding box and name
            if best_match and min_distance < 0.4:  # Use a threshold for recognition
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{best_match}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Real-Time Face Recognition", frame)

        if cv2.waitKey(1) % 256 == 27:  # ESC pressed
            break

    cam.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    while True:
        print("1. Capture new face data")
        print("2. Start real-time face recognition")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            name = input("Enter the name of the person: ")
            capture_face_data(name)
        elif choice == '2':
            recognize_faces()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")
