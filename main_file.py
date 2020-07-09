import cv2
import os
import numpy as np
import functions as func

templates_path = os.getcwd() + '\\Templates\\'
all_cards = func.get_cards(templates_path)
all_templates = func.get_all_templates(all_cards)


video_capture = cv2.VideoCapture(0)
while True:

    ret, frame = video_capture.read()
    frame = func.describe_cards(frame, all_cards, all_templates)
    cv2.imshow('Press ESC key to close', frame)
    
    k = cv2.waitKey(1) & 0xff # press 'esc' to close video capture.
    if k == 27:
       break

video_capture.release()
cv2.destroyAllWindows()


for i in np.arange(1, 9):
    input_path = os.getcwd() + '\\TestCards\\Fig' + str(i) + '.png'
    input_img = cv2.imread(input_path)
    cv2.imshow('', input_img)
    cv2.waitKey(0)

    input_img = func.describe_cards(input_img, all_cards, all_templates)
    cv2.imshow('', input_img)
    cv2.waitKey(0)


cv2.destroyAllWindows()
