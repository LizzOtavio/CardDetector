import cv2
import os
import numpy as np
import pandas as pd
import random

# Get all files from Templates folder


def get_cards(path):
    cards = []
    for file in os.listdir(path):
        if file.endswith(".png"):
            cards.append(path + file)
    return cards

# Get slice of card that represents name and suit


def get_all_templates(cards):
    templates = []
    for card in cards:
        im = cv2.imread(card)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (197, 282))
        template_suit = im.copy()
        template_suit = template_suit[52:85, 12:42]

        template_name = im.copy()
        template_name = template_name[12:52, 12:42]

        name = card.split('\\')[-1].replace('.png', '')
        suit = name.split('_')[-1]
        name = name.split('_')[0]

        templates.append({
            "name": name,
            "suit": suit,
            "template_name": template_name,
            "template_suit": template_suit
        })
    return pd.DataFrame(templates)


def match_templates_to_card(test_img, all_templates):

    templates_suit = all_templates['template_suit'].values
    test_img, index = calcule_template_match(test_img, templates_suit)
    suit = all_templates['suit'].iloc[index]

    templates_name = all_templates['template_name'].values
    test_img, index = calcule_template_match(test_img, templates_name)
    name = all_templates['name'].iloc[index]
    return test_img, name, suit

# apply template match function - the highest max_val represents the best match


def calcule_template_match(im, templates):
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    width, height = templates[0].shape[::-1]
    best_match = []
    index = 0
    for template in templates:
        result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        best_match.append({
            "max_val": max_val,
            "max_loc": max_loc,
            "index": index
        })
        index = index + 1
    df = pd.DataFrame(best_match)
    matched_elem = df[df['max_val'] == df['max_val'].max()]
    max_loc = matched_elem['max_loc'].values[0]
    index = matched_elem['index'].values[0]

    bottom_right = (max_loc[0] + width, max_loc[1] + height)
    R = random.randint(0, 255)
    G = random.randint(0, 255)
    B = random.randint(0, 255)

    cv2.rectangle(im, max_loc, bottom_right, (B, G, R), 1)

    return im, index


def describe_cards(input_img, all_cards, all_templates):
    default_size = cv2.imread(all_cards[0]).shape
    def_height = default_size[0]
    def_width = default_size[1]
    pts_dst = np.array([(0, 0), (0, def_height), (def_width, def_height), (def_width, 0)]).reshape((4, 2))

    blur = cv2.GaussianBlur(input_img, (3, 3), 0)
    edge = cv2.Canny(blur, 50, 200, None, 3)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(edge, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # image is dilated then eroded to avoid double countour
    cards = []
    for i in range(0, len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > 5000: # Filter for 
            cv2.drawContours(input_img, [cnt], 0, (0, 255, 0), 2)
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            try:
                x1 = approx[0][0][0]  # get four points in corners
                y1 = approx[0][0][1]
                x2 = approx[1][0][0]
                y2 = approx[1][0][1]
                x3 = approx[2][0][0]
                y3 = approx[2][0][1]
                x4 = approx[3][0][0]
                y4 = approx[3][0][1]
                # correct pts order -> top-Left (1) - bottom-Left (2) - bottom-right (3) - top-right (4)
                # euclidean distance - Verifies Card Orientation
                dist1 = np.linalg.norm(np.array([x1, y1])-np.array([x2, y2]))
                dist2 = np.linalg.norm(np.array([x1, y1])-np.array([x4, y4]))
                if dist1 > dist2:  # Card is vertical (correct orientation)
                    pts_src = np.array(
                        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).reshape((4, 2))
                else:  # Card is in horizontal -- must rotate before template matching
                    pts_src = np.array(
                        [[x2, y2], [x3, y3], [x4, y4], [x1, y1]]).reshape((4, 2))
                h, status = cv2.findHomography(
                    pts_src, pts_dst)  # correct image perspective
                img = cv2.warpPerspective(
                    input_img, h, (def_width, def_height))
                img, name, suit = match_templates_to_card(img, all_templates)
                ptx = min([x1, x2, x3, x4])
                # get bottom left corner position to insert result
                pty = max([y1, y2, y3, y4])
                cv2.putText(img, '{} of {}'.format(name, suit), (0, 270,),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                cv2.putText(input_img, '{} of {}'.format(
                    name, suit), (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cards.append(img)
            except:
                pass
    return cards, input_img
