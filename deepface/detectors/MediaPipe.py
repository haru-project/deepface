from typing import Any, List
import numpy as np
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from deepface.modules import detection

# Link - https://google.github.io/mediapipe/solutions/face_detection


class MediaPipeClient(Detector):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> Any:
        """
        Build a mediapipe face detector model
        Returns:
            model (Any)
        """
        # this is not a must dependency. do not import it in the global level.
        try:
            import mediapipe as mp
        except ModuleNotFoundError as e:
            raise ImportError(
                "MediaPipe is an optional detector, ensure the library is installed."
                "Please install using 'pip install mediapipe' "
            ) from e

        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
        return face_detection

    def detect_faces(
        self, img: np.ndarray, align: bool = True, expand_percentage: int = 0
    ) -> List[DetectedFace]:
        """
        Detect and align face with mediapipe

        Args:
            img (np.ndarray): pre-loaded image as numpy array

            align (bool): flag to enable or disable alignment after detection (default is True)

            expand_percentage (int): expand detected facial area with a percentage

        Returns:
            results (List[Tuple[DetectedFace]): A list of DetectedFace objects
                where each object contains:

            - img (np.ndarray): The detected face as a NumPy array.

            - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h

            - confidence (float): The confidence score associated with the detected face.
        """
        resp = []

        img_width = img.shape[1]
        img_height = img.shape[0]

        results = self.model.process(img)

        # If no face has been detected, return an empty list
        if results.detections is None:
            return resp

        # Extract the bounding box, the landmarks and the confidence score
        for current_detection in results.detections:
            (confidence,) = current_detection.score

            bounding_box = current_detection.location_data.relative_bounding_box
            landmarks = current_detection.location_data.relative_keypoints

            x = int(bounding_box.xmin * img_width)
            w = int(bounding_box.width * img_width)
            y = int(bounding_box.ymin * img_height)
            h = int(bounding_box.height * img_height)

            # Extract landmarks
            left_eye = (int(landmarks[0].x * img_width), int(landmarks[0].y * img_height))
            right_eye = (int(landmarks[1].x * img_width), int(landmarks[1].y * img_height))
            # nose = (int(landmarks[2].x * img_width), int(landmarks[2].y * img_height))
            # mouth = (int(landmarks[3].x * img_width), int(landmarks[3].y * img_height))
            # right_ear = (int(landmarks[4].x * img_width), int(landmarks[4].y * img_height))
            # left_ear = (int(landmarks[5].x * img_width), int(landmarks[5].y * img_height))

            if x > 0 and y > 0:

                # expand the facial area to be extracted and stay within img.shape limits
                x2 = max(0, x - int((w * expand_percentage) / 100))  # expand left
                y2 = max(0, y - int((h * expand_percentage) / 100))  # expand top
                w2 = min(img.shape[1], w + int((w * expand_percentage) / 100))  # expand right
                h2 = min(img.shape[0], h + int((h * expand_percentage) / 100))  # expand bottom

                # detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
                detected_face = img[int(y2) : int(y2 + h2), int(x2) : int(x2 + w2)]

                img_region = FacialAreaRegion(x=x, y=y, w=w, h=h)

                if align:
                    detected_face = detection.align_face(
                        img=detected_face, left_eye=left_eye, right_eye=right_eye
                    )

                detected_face_obj = DetectedFace(
                    img=detected_face,
                    facial_area=img_region,
                    confidence=confidence,
                )

                resp.append(detected_face_obj)

        return resp
