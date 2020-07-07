import cv2
import os
import numpy as np
import functions as func

templates_path = os.getcwd() + '\\Templates\\'
all_cards = func.get_cards(templates_path)
all_templates = func.get_all_templates(all_cards)


for i in np.arange(1, 9):
    input_path = os.getcwd() + '\\TestCards\\Fig' + str(i) + '.png'
    input_img = cv2.imread(input_path)
    cv2.imshow('', input_img)
    cv2.waitKey(0)
    
    cards, input_img = func.describe_cards(input_img, all_cards, all_templates)
    #for card in cards:
    #    cv2.imshow('', card)
    #    cv2.waitKey(0)  
    
    cv2.imshow('', input_img)
    cv2.waitKey(0)


cv2.destroyAllWindows()
