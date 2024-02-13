# built-in dependencies
import os
import pickle
from typing import List, Union, Optional
import time
import concurrent.futures

# 3rd party dependencies
import numpy as np
import pandas as pd
from tqdm import tqdm

# project dependencies
from deepface.commons.logger import Logger
from deepface.modules import representation, detection, modeling, verification
from deepface.models.FacialRecognition import FacialRecognition
from deepface.extendedmodels import Gender, Race, Emotion


logger = Logger(module="deepface/modules/recognition.py")

class faceData:
    def __init__(self, bounding_box, distance_to_match=0.0, name="Unknown", analysis = {}):
        self.bounding_box = bounding_box
        self.name = name
        self.distance_to_match = distance_to_match
        self.analysis = analysis


def process_source_obj(source_obj, model_name, distance_metric, df, threshold, actions, silent, enforce_detection, align, normalization):
    source_img = source_obj["face"]
    source_region = source_obj["facial_area"]
    source_confidence = source_obj["confidence"]

    face = faceData(bounding_box=[source_region["x"], source_region["y"], source_region["w"], source_region["h"]])

    target_embedding_obj = representation.represent(
        img_path=source_img,
        model_name=model_name,
        enforce_detection=enforce_detection,
        detector_backend="skip",
        align=align,
        normalization=normalization,
    )

    target_representation = target_embedding_obj[0]["embedding"]

    result_df = df.copy()
    result_df["source_x"] = source_region["x"]
    result_df["source_y"] = source_region["y"]
    result_df["source_w"] = source_region["w"]
    result_df["source_h"] = source_region["h"]

    distances = []
    for _, instance in df.iterrows():
        source_representation = instance[f"{model_name}_representation"]

        target_dims = len(list(target_representation))
        source_dims = len(list(source_representation))
        if target_dims != source_dims:
            raise ValueError("Source and target embeddings must have the same dimensions.")

        if distance_metric == "cosine":
            distance = verification.find_cosine_distance(source_representation, target_representation)
        elif distance_metric == "euclidean":
            distance = verification.find_euclidean_distance(source_representation, target_representation)
        elif distance_metric == "euclidean_l2":
            distance = verification.find_euclidean_distance(
                verification.l2_normalize(source_representation),
                verification.l2_normalize(target_representation),
            )
        else:
            raise ValueError(f"Invalid distance metric passed - {distance_metric}")

        distances.append(distance)

    target_threshold = threshold or verification.find_threshold(model_name, distance_metric)

    result_df["threshold"] = target_threshold
    result_df["distance"] = distances

    result_df = result_df.drop(columns=[f"{model_name}_representation"])
    result_df = result_df.sort_values(by=["distance"], ascending=True).reset_index(drop=True)

    if float(result_df.loc[0, "distance"]) < target_threshold:
        face.name = result_df.loc[0, "identity"]
        face.distance_to_match = result_df.loc[0, "distance"]

    if source_img.shape[0] > 0 and source_img.shape[1] > 0:
        obj = {}
        # facial attribute analysis
        pbar = tqdm(
            range(0, len(actions)),
            desc="Finding actions",
            disable=silent if len(actions) > 1 else True,
        )
        for index in pbar:
            action = actions[index]
            pbar.set_description(f"Action: {action}")
            if action == "emotion":
                emotion_predictions = modeling.build_model("Emotion").predict(source_img)
                sum_of_predictions = emotion_predictions.sum()
                obj["emotion"] = {}
                for i, emotion_label in enumerate(Emotion.labels):
                    emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                    obj["emotion"][emotion_label] = emotion_prediction
                obj["dominant_emotion"] = Emotion.labels[np.argmax(emotion_predictions)]
            elif action == "age":
                apparent_age = modeling.build_model("Age").predict(source_img)
                # int cast is for exception - object of type 'float32' is not JSON serializable
                obj["age"] = int(apparent_age)
            elif action == "gender":
                gender_predictions = modeling.build_model("Gender").predict(source_img)
                obj["gender"] = {}
                for i, gender_label in enumerate(Gender.labels):
                    gender_prediction = 100 * gender_predictions[i]
                    obj["gender"][gender_label] = gender_prediction
                obj["dominant_gender"] = Gender.labels[np.argmax(gender_predictions)]
            elif action == "race":
                race_predictions = modeling.build_model("Race").predict(source_img)
                sum_of_predictions = race_predictions.sum()
                obj["race"] = {}
                for i, race_label in enumerate(Race.labels):
                    race_prediction = 100 * race_predictions[i] / sum_of_predictions
                    obj["race"][race_label] = race_prediction
                obj["dominant_race"] = Race.labels[np.argmax(race_predictions)]
            # -----------------------------
            # mention facial areas
            obj["region"] = source_region
            # include image confidence
            obj["face_confidence"] = source_confidence
            obj["face"] = source_img
        face.analysis = obj
    return face

def strawberry_find(
    img_path: Union[str, np.ndarray],
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    threshold: Optional[float] = None,
    normalization: str = "base",
    silent: bool = False,
    actions: Union[tuple, list] = ("emotion", "gender"),
    #actions: Union[tuple, list] = ("emotion", "age", "gender", "race"),
) -> List[pd.DataFrame]:
    """
    Identify individuals in a database

    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2'.

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8'.

        align (boolean): Perform alignment based on the eye positions.

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

        silent (boolean): Suppress or allow some log messages for a quieter analysis process.
        
        actions (tuple): Attributes to analyze. The default is ('age', 'gender', 'emotion', 'race').
            You can exclude some of these attributes from the analysis if needed.
    
    Returns:
        results (List[pd.DataFrame]): A list of pandas dataframes. Each dataframe corresponds
            to the identity information for an individual detected in the source image.
            The DataFrame columns include:

            - 'identity': Identity label of the detected individual.

            - 'target_x', 'target_y', 'target_w', 'target_h': Bounding box coordinates of the
                    target face in the database.

            - 'source_x', 'source_y', 'source_w', 'source_h': Bounding box coordinates of the
                    detected face in the source image.

            - 'threshold': threshold to determine a pair whether same person or different persons

            - 'distance': Similarity score between the faces based on the
                    specified model and distance metric


            XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

            UPDATE

            XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    """

    tic = time.time()

    # -------------------------------
    if os.path.isdir(db_path) is not True:
        raise ValueError("Passed db_path does not exist!")

    model: FacialRecognition = modeling.build_model(model_name)
    target_size = model.input_shape

    # ---------------------------------------

    file_name = f"representations_{model_name}.pkl"
    file_name = file_name.replace("-", "_").lower()
    datastore_path = f"{db_path}/{file_name}"

    df_cols = [
        "identity",
        f"{model_name}_representation",
        "target_x",
        "target_y",
        "target_w",
        "target_h",
    ]

    if os.path.exists(datastore_path):
        with open(datastore_path, "rb") as f:
            representations = pickle.load(f)

            if len(representations) > 0 and len(representations[0]) != len(df_cols):
                raise ValueError(
                    f"Seems existing {datastore_path} is out-of-the-date."
                    "Please delete it and re-run."
                )

        alpha_employees = __list_images(path=db_path)
        beta_employees = [representation[0] for representation in representations]

        newbies = list(set(alpha_employees) - set(beta_employees))
        oldies = list(set(beta_employees) - set(alpha_employees))

        if newbies:
            logger.warn(
                f"Items {newbies} were added into {db_path}"
                f" just after data source {datastore_path} created!"
            )
            newbies_representations = __find_bulk_embeddings(
                employees=newbies,
                model_name=model_name,
                target_size=target_size,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=align,
                normalization=normalization,
                silent=silent,
            )
            representations = representations + newbies_representations

        if oldies:
            logger.warn(
                f"Items {oldies} were dropped from {db_path}"
                f" just after data source {datastore_path} created!"
            )
            representations = [rep for rep in representations if rep[0] not in oldies]

        if newbies or oldies:
            if len(representations) == 0:
                raise ValueError(f"There is no image in {db_path} anymore!")

            # save new representations
            with open(datastore_path, "wb") as f:
                pickle.dump(representations, f)

            if not silent:
                logger.info(
                    f"{len(newbies)} new representations are just added"
                    f" whereas {len(oldies)} represented one(s) are just dropped"
                    f" in {db_path}/{file_name} file."
                )

        if not silent:
            logger.info(f"There are {len(representations)} representations found in {file_name}")

    else:  # create representation.pkl from scratch
        employees = __list_images(path=db_path)

        if len(employees) == 0:
            raise ValueError(
                f"There is no image in {db_path} folder!"
                "Validate .jpg, .jpeg or .png files exist in this path.",
            )

        # ------------------------
        # find representations for db images
        representations = __find_bulk_embeddings(
            employees=employees,
            model_name=model_name,
            target_size=target_size,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            normalization=normalization,
            silent=silent,
        )

        # -------------------------------

        with open(datastore_path, "wb") as f:
            pickle.dump(representations, f)

        if not silent:
            logger.info(f"Representations stored in {db_path}/{file_name} file.")

    # ----------------------------
    # now, we got representations for facial database
    df = pd.DataFrame(
        representations,
        columns=df_cols,
    )

    # img path might have more than once face
    source_objs = detection.extract_faces(
        img_path=img_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
    )

    resp_obj = []
    faces = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_source_obj, source_obj, model_name, distance_metric, df, threshold, actions, silent, enforce_detection, align, normalization)
                   for source_obj in source_objs]

        for future in concurrent.futures.as_completed(futures):
            faces.append(future.result())
    # -----------------------------------

    toc = time.time()

    if not silent:
        logger.info(f"find function lasts {toc - tic} seconds")

    return faces


def __list_images(path: str) -> list:
    """
    List images in a given path
    Args:
        path (str): path's location
    Returns:
        images (list): list of exact image paths
    """
    images = []
    for r, _, f in os.walk(path):
        for file in f:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                exact_path = f"{r}/{file}"
                images.append(exact_path)
    return images


def __find_bulk_embeddings(
    employees: List[str],
    model_name: str = "VGG-Face",
    target_size: tuple = (224, 224),
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
):
    """
    Find embeddings of a list of images

    Args:
        employees (list): list of exact image paths

        model_name (str): facial recognition model name

        target_size (tuple): expected input shape of facial recognition model

        detector_backend (str): face detector model name

        enforce_detection (bool): set this to False if you
            want to proceed when you cannot detect any face

        align (bool): enable or disable alignment of image
            before feeding to facial recognition model

        expand_percentage (int): expand detected facial area with a
            percentage (default is 0).

        normalization (bool): normalization technique

        silent (bool): enable or disable informative logging
    Returns:
        representations (list): pivot list of embeddings with
            image name and detected face area's coordinates
    """
    representations = []
    for employee in tqdm(
        employees,
        desc="Finding representations",
        disable=silent,
    ):
        img_objs = detection.extract_faces(
            img_path=employee,
            target_size=target_size,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
        )

        for img_obj in img_objs:
            img_content = img_obj["face"]
            img_region = img_obj["facial_area"]
            embedding_obj = representation.represent(
                img_path=img_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            img_representation = embedding_obj[0]["embedding"]

            instance = []
            instance.append(employee)
            instance.append(img_representation)
            instance.append(img_region["x"])
            instance.append(img_region["y"])
            instance.append(img_region["w"])
            instance.append(img_region["h"])
            representations.append(instance)
    return representations
