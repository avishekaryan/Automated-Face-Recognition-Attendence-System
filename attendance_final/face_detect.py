"""
face_detect.py
---------------
Just the detection half of face recognition: "is there a face in this
frame, and where." Uses OpenCV's Haar Cascade - a classical, fast
algorithm that slides a chain of simple brightness-pattern checks across
the image, rejecting obviously-not-a-face regions almost instantly and
only spending real work on regions that look promising.

This module is imported by BOTH sides of the network boundary:
    - server.py / face_recognizer.py use it as the first step of
      actually recognizing someone (detect, then run LBPH on the result).
    - client_gui.py uses it too, but only to answer a much narrower
      question locally: "is anyone even in frame right now?" Detection
      alone can never say WHO a face belongs to - only "a face-shaped
      region exists here" - so letting the client run this doesn't hand
      it any ability to identify people. That stays on the server, where
      the trained model and the actual recognize() call live.

Keeping this in one shared file (instead of copy-pasted in two places)
means there's exactly one cascade to load and one set of detection
parameters to tune.
"""

import os
import cv2

# Load the cascade from a copy shipped directly next to this file, rather
# than relying on cv2.data.haarcascades - on some installs (notably some
# Anaconda setups on Windows) that folder is missing its data files even
# though the cv2 package itself is installed correctly. Bundling our own
# copy means this works the same way on every machine, regardless of
# whether that particular install issue is present.
_LOCAL_CASCADE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "haarcascade_frontalface_default.xml")

if os.path.exists(_LOCAL_CASCADE):
    _CASCADE_PATH = _LOCAL_CASCADE
else:
    _CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

_detector = cv2.CascadeClassifier(_CASCADE_PATH)
if _detector.empty():
    raise RuntimeError(
        f"Could not load the Haar Cascade file from '{_CASCADE_PATH}'. "
        "Make sure haarcascade_frontalface_default.xml is present next to "
        "face_detect.py, or that your OpenCV install's data files are intact."
    )


def detect_faces(gray_frame):
    """Returns a list of (x, y, w, h) bounding boxes for every face-shaped
    region found. Expects a grayscale image - both Haar Cascade and LBPH
    work on grayscale only."""
    return _detector.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)
