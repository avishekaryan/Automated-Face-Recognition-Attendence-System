"""
face_recognizer.py
-------------------
Runs only on the server - one trained model, one source of truth. Uses
LBPH: compares face images/histograms directly, no embeddings or
identity vectors stored.

Dataset layout: one subfolder per student -
    dataset/student_1/1.jpg, dataset/student_1/2.jpg, ...
    dataset/student_2/1.jpg, ...
rather than every student's images sitting flat in one folder with the
ID embedded in the filename. This makes browsing/searching the raw
dataset on disk straightforward (open one folder, see one student's
photos) instead of having to pick a specific student's files out of a
folder shared by everyone.

Two different inputs arrive here, handled by two different methods:

    - recognize_crop(): the splash screen's continuous recognition loop.
      The CLIENT already ran local Haar Cascade detection (see
      face_detect.py) before sending anything, and only sends a frame
      when it found a face - already cropped down to roughly that face
      region. So this method does NOT re-run detection; it trusts the
      crop and goes straight to LBPH prediction. This is what keeps the
      constant splash-screen traffic cheap: empty frames never get sent
      at all, and what does get sent is already small.

    - save_sample(): used only during Admin registration, which is rare
      (a few dozen calls per new student, not once-a-second-forever).
      Here the server DOES run its own detection, because the client's
      capture loop needs a face-found/not-found answer back to know
      whether to keep the frame - detection is genuinely part of the
      registration protocol's feedback loop, not just a filter.
"""

import base64
import os
import re

import cv2
import numpy as np

from face_detect import detect_faces

DATASET_DIR = "dataset"
MODEL_PATH = "trainer.yml"
CONFIDENCE_THRESHOLD = 70  # lower LBPH distance = more confident match

_STUDENT_FOLDER_RE = re.compile(r"^student_(\d+)$")


def _student_folder(student_id: int) -> str:
    path = os.path.join(DATASET_DIR, f"student_{student_id}")
    os.makedirs(path, exist_ok=True)
    return path


def decode_frame(image_b64: str):
    """Turns a base64 JPEG (as sent by a client) into an OpenCV BGR image."""
    data = base64.b64decode(image_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class FaceRecognizer:
    def __init__(self):
        os.makedirs(DATASET_DIR, exist_ok=True)
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._trained = False
        if os.path.exists(MODEL_PATH):
            self._recognizer.read(MODEL_PATH)
            self._trained = True

    def save_sample(self, student_id: int, image_b64: str) -> int:
        """Saves one training frame for a student, into that student's own
        subfolder. Runs its own detection since the caller needs a
        face-found/not-found answer back. Returns the sample count so far
        for this student, or -1 if no face was found."""
        frame = decode_frame(image_b64)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        if len(faces) == 0:
            return -1
        x, y, w, h = faces[0]
        face = gray[y:y + h, x:x + w]
        folder = _student_folder(student_id)
        existing = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        count = len(existing) + 1
        cv2.imwrite(os.path.join(folder, f"{count}.jpg"), face)
        return count

    
    def clear_samples(self, student_id: int) -> int:
        """Deletes every saved face image for one student (used before a re-capture)."""
        folder = _student_folder(student_id)
        removed = 0
        for f in os.listdir(folder):
            if f.endswith(".jpg"):
                os.remove(os.path.join(folder, f))
                removed += 1
        return removed
    
    def train(self):
        faces, ids = [], []
        for entry in os.listdir(DATASET_DIR):
            entry_path = os.path.join(DATASET_DIR, entry)
            if not os.path.isdir(entry_path):
                continue
            match = _STUDENT_FOLDER_RE.match(entry)
            if not match:
                continue
            student_id = int(match.group(1))
            for filename in os.listdir(entry_path):
                if not filename.endswith(".jpg"):
                    continue
                image = cv2.imread(os.path.join(entry_path, filename), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                faces.append(np.array(image, dtype="uint8"))
                ids.append(student_id)

        if not faces:
            raise ValueError("No training samples found - register a student first.")

        self._recognizer.train(faces, np.array(ids))
        self._recognizer.write(MODEL_PATH)
        self._trained = True
        return len(faces), len(set(ids))

    def recognize_crop(self, image_b64: str):
        """Given an already-cropped face image (the client did detection),
        returns (student_id, confidence) or (None, None) if untrained or
        no confident match. No detection step here - see module docstring."""
        if not self._trained:
            return None, None
        frame = decode_frame(image_b64)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        student_id, distance = self._recognizer.predict(gray)
        if distance < CONFIDENCE_THRESHOLD:
            return student_id, distance
        return None, distance
