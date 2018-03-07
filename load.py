
import pickle
# for visualize
import cv2

import sys
import time

with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)

for e in data:
    s = e[0]
    s = cv2.cvtColor(s, cv2.COLOR_RGB2BGR)
    cv2.imshow('observation', s)
    cv2.waitKey(2)
