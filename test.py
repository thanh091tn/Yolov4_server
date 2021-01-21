import os
import colorsys

import numpy as np
from tensorflow.compat.v1.keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import cv2
import pytesseract
import re
from reco import E2E
from pathlib import Path
import argparse
import time


class Yolo4(object):

    def detect_licence(self, img, coords):
        LpImg1 = img.crop(coords)
        #
        LpImg1.save('cr.jpg')
        img1 = cv2.imread('cr.jpg')

        model = E2E()

        rs = model.predict(img1)

        gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        # resize image to three times as large as original for better readability
        gray = cv2.resize(gray, None, fx=3, fy=3,
                          interpolation=cv2.INTER_CUBIC)

        # perform gaussian blur to smoothen image
        # blur = cv2.GaussianBlur(gray, (5, 5), 0)
        blur = cv2.medianBlur(gray, 5)
        cv2.imwrite('cc.jpg', blur)
        # cv2.imshow("Gray", gray)
        # cv2.waitKey(0)
        # threshold the image using Otsus method to preprocess for tesseract
        ret, thresh = cv2.threshold(blur, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow("Otsu Threshold", thresh)
        # cv2.waitKey(0)
        # create rectangular kernel for dilation
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        # apply dilation to make regions more clear
        dilation = cv2.dilate(thresh, rect_kern, iterations=1)
        cv2.imwrite('ca.jpg', dilation)
        # cv2.imshow("Dilation", dilation)
        # cv2.waitKey(0)
        # find contours of regions of interest within license plate
        # try:
        #     contours, hierarchy = cv2.findContours(
        #         dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # except:
        #     ret_img, contours, hierarchy = cv2.findContours(
        #         dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # sort contours left-to-right
        # sorted_contours = sorted(
        #     contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        # # create copy of gray image
        # im2 = dilation.copy()
        # # create blank string to hold license plate number
        # plate_num = ""
        # # loop through contours and find individual letters and numbers in license plate
        # for cnt in sorted_contours:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     height, width = im2.shape
        #     # if height of box is not tall enough relative to total height then skip
        #     if height / float(h) > 6:
        #         continue

        #     ratio = h / float(w)
        #     # if height to width ratio is less than 1.5 skip
        #     if ratio < 1.5:
        #         continue

        #     # if width is not wide enough relative to total width then skip
        #     if width / float(w) > 15:
        #         continue

        #     area = h * w
        #     # if area is less than 100 pixels skip
        #     if area < 100:
        #         continue

        #     # draw the rectangle
        #     rect = cv2.rectangle(im2, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #     # grab character region of image
        #     roi = dilation[y-5:y+h+5, x-5:x+w+5]

        #     # perfrom bitwise not to flip image to black text on white background
        #     #roi = cv2.bitwise_not(roi)

        #     # perform another blur on character region
        #     roi = cv2.resize(roi, None, fx=3, fy=3,
        #                   interpolation=cv2.INTER_CUBIC)
        #     roi = cv2.GaussianBlur(roi, (3,3), 0)
        #     cv2.imwrite('roi.jpg',roi)
        #     try:
        #         text = pytesseract.image_to_string(
        #             roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
        #         # clean tesseract text by removing any unwanted blank spaces
        #         clean_text = re.sub('[\W_]+', '', text)

        #         plate_num += clean_text
        #         print(plate_num)
        #     except Exception as e:
        #         print(e)
        #         text = None
        t = pytesseract.image_to_string(
            dilation, lang="eng", config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6 --oem 3')
        rs = re.sub('[\W_]+', '', t)
        print(rs)
        # if plate_num != None:
        #     print("License Plate #: ", plate_num)
        # cv2.imshow("Character's Segmented", im2)
        # cv2.waitKey(0)
        return rs

    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file.'

        self.class_names = self.get_class()
        self.anchors = self.get_anchors()

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        self.sess = K.get_session()

        # Load model, or construct model and load weights.
        self.yolo4_model = yolo4_body(
            Input(shape=(608, 608, 3)), num_anchors//3, num_classes)
        self.yolo4_model.load_weights(model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        if self.gpu_num >= 2:
            self.yolo4_model = multi_gpu_model(
                self.yolo4_model, gpus=self.gpu_num)

        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo4_model.output, self.anchors,
                                                          len(self.class_names), self.input_image_shape,
                                                          score_threshold=self.score)

    def __init__(self, score, iou, anchors_path, classes_path, model_path, gpu_num=1):
        self.score = score
        self.iou = iou
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.model_path = model_path
        self.gpu_num = gpu_num
        self.load_yolo()

    def close_session(self):
        self.sess.close()

    def detect_image(self, image, model_image_size=(608, 608)):
        start = timer()
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo4_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                # K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        listPositions = {}
        listLaybel = {}
        p = {}
        ll = []
        la = []
        z = 0
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{}   {:.2f}%'.format(predicted_class, score*100)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            sc = '{:.2f}%'.format(score*100)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            name = label
            rs = (predicted_class, left, top, right, bottom)
            p = {
                "left": int(left),
                "top": int(top),
                "right": int(right),
                "bottom": int(bottom)
            }
            # listPositions['item'+str(z)] = {
            #     'name': name
            # }
            z += 1
            listLaybel['item'+str(z)] = name
            ll.append({'name': predicted_class,
                       'score': sc, 'position': p})

            # cv2.rectangle(image, (int(left), int(top)),
            #               (int(right), int(bottom)), (0, 255, 0), 5)
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        image.save('img/abc.jpg')
        print(Path().absolute())
        end = timer()

        return ll


if __name__ == '__main__':
    # Duong dan den file h5
    model_path = 'aa.h5'
    # File anchors cua YOLO
    anchors_path = 'yolo4_anchors.txt'
    # File danh sach cac class
    classes_path = 'yolo.names'

    score = 0.5
    iou = 0.5

    model_image_size = (608, 608)

    img = "a.jpg"
    image = Image.open(img)
    yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path)
    result = yolo4_model.detect_image(image, model_image_size=model_image_size)
    plt.imshow(result)
    plt.show()
    yolo4_model.close_session()
