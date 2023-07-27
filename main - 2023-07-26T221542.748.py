import face_recognition
import cv2

def load_known_faces(known_faces_dir):
    known_faces = []
    for filename in os.listdir(known_faces_dir):
        image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        face_encoding = face_recognition.face_encodings(image)[0]
        known_faces.append((face_encoding, filename.split('.')[0]))
    return known_faces

def recognize_faces(known_faces, unknown_image_path):
    unknown_image = face_recognition.load_image_file(unknown_image_path)
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)

    for unknown_face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces([face[0] for face in known_faces], unknown_face_encoding)

        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = known_faces[match_index][1]

        print(f"Found face: {name}")

        # Draw a rectangle around the face
        top, right, bottom, left = face_recognition.face_locations(unknown_image)[0]
        cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(unknown_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Facial Recognition", unknown_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    known_faces_dir = "known_faces"
    unknown_image_path = "unknown_image.jpg"

    known_faces = load_known_faces(known_faces_dir)
    recognize_faces(known_faces, unknown_image_path)
