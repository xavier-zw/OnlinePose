import cv2
import numpy as np
import torch
from torch import nn


class Detect(nn.Module):
    def __init__(self):
        super(Detect, self).__init__()
        self.net = cv2.dnn.readNetFromDarknet(r"D:\Jupter_book\Qt\onlinepose\yolov3-tiny.cfg",
                                              r"D:\Jupter_book\Qt\onlinepose\yolov3-tiny.weights")
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [x.strip() for x in f.readlines()]
        layer_name = self.net.getLayerNames()
        self.output_layers = [layer_name[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def forward(self, img):
        height, width, c = img.shape
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
        class_ids = []
        confidences = []
        boxes = []
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  #只要人
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        return [boxes[i] for i in indexes]