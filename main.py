from face import compare_faces, load_images
from ultralytics import YOLO
import cv2


def resize_image(img, new_width, new_height):
    final_width = new_width
    final_height = new_height

    height, width = img.shape[:2]

    # Calculate ratio of new width and height with respect to old width and height
    ratio_width = new_width / width
    ratio_height = new_height / height

    # Get minimum ratio to maintain aspect ratio
    ratio = min(ratio_width, ratio_height)

    # Calculate new dimensions to maintain aspect ratio
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Resize image to new dimensions
    resized_img = cv2.resize(img, (new_width, new_height))

    # Calculate borders to add
    top_border = (final_height - height) // 2
    bottom_border = final_height - height - top_border
    left_border = (final_width - width) // 2
    right_border = final_width - width - left_border

    # Add borders
    img_with_borders = cv2.copyMakeBorder(
        resized_img,
        top_border,
        bottom_border,
        left_border,
        right_border,
        cv2.BORDER_CONSTANT,
        None,
        value=(0, 0, 0),
    )

    # final = cv2.resize(img_with_borders, (final_width, final_height))
    # return final

    final_img = cv2.resize(img_with_borders, (final_width, final_height))

    return final_img


def detect():
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    results = model(
        0,
        show=True,
        classes=[0],
        stream=True,
        save=True,
    )  # predict on an image

    for result in results:
        final_img_coordinates = [float("inf"), float("inf"), 0, 0]

        for box in result.boxes:
            xyxy = list(map(int, box.xyxy[0]))
            if xyxy[0] < final_img_coordinates[0]:
                final_img_coordinates[0] = xyxy[0]
            if xyxy[1] < final_img_coordinates[1]:
                final_img_coordinates[1] = xyxy[1]
            if xyxy[2] > final_img_coordinates[2]:
                final_img_coordinates[2] = xyxy[2]
            if xyxy[3] > final_img_coordinates[3]:
                final_img_coordinates[3] = xyxy[3]

        final_img_coordinates_predicates = list(
            map(lambda x: x == float("inf"), final_img_coordinates)
        )

        is_no_object_detected = any(final_img_coordinates_predicates)
        if is_no_object_detected:
            final_img_coordinates = [
                0,
                0,
                result.orig_img.shape[1],
                result.orig_img.shape[0],
            ]
        cropped = result.orig_img[
            final_img_coordinates[1] : final_img_coordinates[3],
            final_img_coordinates[0] : final_img_coordinates[2],
        ]

        results = compare_faces(cropped)
        print(results)
        resized = None
        if is_no_object_detected:
            resized = cropped
        else:
            resized = resize_image(cropped, 640, 480)

        cv2.imshow("cropped", resized)
        # cv2.waitKey(0)


load_images()
detect()

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



