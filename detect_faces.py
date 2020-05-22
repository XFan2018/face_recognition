import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt


def get_model(prototxt_dir, caffemodel_dir):
    """
    :param prototxt_dir: prototxt path
    :param caffemodel_dir: caffemodel path
    :return: model
    """
    model = cv2.dnn.readNetFromCaffe(prototxt_dir, caffemodel_dir)
    return model


def detect_faces(dataset_path, dataset_dir_list, prototxt_dir, caffemodel_dir, confidence_threshold=0.5):
    """
    :param dataset_path         dataset path
    :param dataset_dir_list:         dataset file list
    :param prototxt_dir:        prototxt path
    :param caffemodel_dir:      caffemodel path
    :param confidence_threshold: confidence threshold
    :return:                    list of extracted faces
    """
    model = get_model(prototxt_dir, caffemodel_dir)
    result = []
    for file in sorted(dataset_dir_list):
        file_name, file_extension = os.path.splitext(file)
        if file_extension == ".png":
            print(file)
        img = cv2.imread(dataset_path + "/" + file)
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        model.setInput(blob)
        detections = model.forward()
        # detection[0,0,i,2]: confidence
        # detection[0,0,i,3]: leftBottom
        # detection[0,0,i,4]: rightBottom
        # detection[0,0,i,5]: leftTop
        # detection[0,0,i,6]: rightTop
        # i is index of detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (leftBottom, rightBottom, leftTop, rightTop) = box.astype("int")
                text = f"confidence:{confidence * 100:.2f}"
                if rightTop - rightBottom < 120 or leftBottom - rightBottom < 120:
                    continue
                faceROI = img[rightBottom:rightTop, leftBottom:leftTop]
                (fh, fw) = faceROI.shape[:2]
                cv2.rectangle(img, (leftBottom, rightBottom), (leftTop, rightTop), (255, 0, 0), 2)
                cv2.putText(img, text, (leftBottom, rightBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        faceblob = cv2.dnn.blobFromImage(faceROI, 1.0 / 255, (112, 112), (0, 0, 0), swapRB=True, crop=False)
        faceblob = torch.from_numpy(faceblob)
        result.append(faceblob)
        print(faceblob.shape)
    result = tuple(result)
    result = torch.cat(result, 0)
    print("result length: ", len(result))
    return result
