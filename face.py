import face_recognition
import os
import numpy as np

known_encodings = []
known_names = []
folder_path = r"C:\Projects\zoom-in-out\faces"


def load_images():
    global known_encodings, folder_path, known_names
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            known_image = face_recognition.load_image_file(file_path)
            encoding = face_recognition.face_encodings(known_image)[0]
            known_encodings.append(encoding)
            known_names.append(filename.split(".")[0])

    return


def compare_faces(unknown_image):
    global known_encodings
    results = []
    unknown_encodings = face_recognition.face_encodings(unknown_image)
    for unknown_encoding in unknown_encodings:
        matches = face_recognition.compare_faces(known_encodings, unknown_encoding)
        face_distances = face_recognition.face_distance(
            known_encodings, unknown_encoding
        )
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]
            results.append(name)

    return results



