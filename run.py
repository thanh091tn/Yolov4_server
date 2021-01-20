from reco import E2E
import cv2
from pathlib import Path
import argparse
import time


img = cv2.imread('cr.jpg')

model = E2E()

rs = model.predict(img)