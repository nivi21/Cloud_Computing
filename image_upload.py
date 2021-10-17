# import the necessary packages
import numpy as np
import sys
import time
import cv2
import os
# from flask import Flask, request, jsonify, make_response
import json
import base64
import boto3
import traceback

# construct the argument parse and parse the arguments
confthres = 0.3
nmsthres = 0.1


def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    lpath = os.path.sep.join([yolo_path, labels_path])

    print(yolo_path)
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS


def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath


def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath


def load_model(configpath, weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def do_prediction(image, net, LABELS):
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # ensure at least one detection exists
    # returns a list with the parameters recognised in the figure
    objects = []
    detection = {}
    detection['rectangle'] = {}
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            detection['label'] = format(LABELS[classIDs[i]])
            detection['accuracy'] = float(format(confidences[i]))
            detection['rectangle']['height'] = int(format(boxes[i][3]))
            detection['rectangle']['width'] = int(format(boxes[i][2]))
            detection['rectangle']['top'] = int(format(boxes[i][1]))
            detection['rectangle']['left'] = int(format(boxes[i][0]))
            """print("detected item:{}, accuracy:{}, X:{}, Y:{}, width:{}, height:{}".format(LABELS[classIDs[i]],
                                                                                             confidences[i],
                                                                                             boxes[i][0],
                                                                                             boxes[i][1],
                                                                                             boxes[i][2],
                                                                                             boxes[i][3]))"""
            if (detection['accuracy'] > 0.5):
                objects.append(detection.copy())
    return (objects)  # returns  an empty list of no parameters are recognised


## argument
# if len(sys.argv) != 2:
#     raise ValueError("Argument list is wrong. Please use the following format:  {} {}".
#                      format("python iWebLens_server.py", "<yolo_config_folder>"))

# import the necessary packages
import numpy as np
import sys
import time
import cv2
import os
# from flask import Flask, request, jsonify, make_response
import json
import base64
import boto3

# construct the argument parse and parse the arguments
confthres = 0.3
nmsthres = 0.1


def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    lpath = os.path.sep.join([yolo_path, labels_path])

    print(yolo_path)
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS


def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath


def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath


def load_model(configpath, weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def do_prediction(image, net, LABELS):
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # ensure at least one detection exists
    # returns a list with the parameters recognised in the figure
    objects = []
    detection = {}
    detection['rectangle'] = {}
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            detection['label'] = format(LABELS[classIDs[i]])
            detection['accuracy'] = float(format(confidences[i]))
            # detection['rectangle']['height'] = int(format(boxes[i][3]))
            # detection['rectangle']['width'] = int(format(boxes[i][2]))
            # detection['rectangle']['top'] = int(format(boxes[i][1]))
            # detection['rectangle']['left'] = int(format(boxes[i][0]))
            # """print("detected item:{}, accuracy:{}, X:{}, Y:{}, width:{}, height:{}".format(LABELS[classIDs[i]],
            #                                                                                  confidences[i],
            #                                                                                  boxes[i][0],
            #                                                                                  boxes[i][1],
            #                                                                                  boxes[i][2],
            #                                                                                  boxes[i][3]))"""
            if (detection['accuracy'] > 0.5):
                objects.append(detection['label'])
    return (objects)  # returns  an empty list of no parameters are recognised


s3_client = boto3.client('s3')
dynamodb=boto3.resource('dynamodb')

labelsPath = s3_client.get_object(Bucket='tagtag-yolo-config', Key='coco.names')
cfgpath = s3_client.get_object(Bucket='tagtag-yolo-config', Key='yolov3-tiny.cfg')
wpath = s3_client.get_object(Bucket='tagtag-yolo-config', Key='yolov3-tiny.weights')

Lables = labelsPath['Body'].read().decode('utf8').strip().split("\n")
CFG = cfgpath['Body'].read()
Weights = wpath['Body'].read()

def lambda_handler(event, context):
    try:
        response_dict = {}
        id=[];
        print(event)
        #event1=json.loads(event)
        print((event["body"]))
        imagefile = json.loads(event["body"])
        print(imagefile["id"])
        print(imagefile["id"].split("data:image/jpeg;base64,")[-1])
        imagefile = base64.b64decode(imagefile["id"].split("data:image/jpeg;base64,")[-1].encode('utf-8'))
        npimg = np.frombuffer(imagefile, np.uint8)  # Interpret a buffer as a 1-dimensional array
        img = cv2.imdecode(npimg,-1)  # The function imdecode reads an image from the specified buffer in the memory.
        npimg = np.array(img)  # convert the read image to a numpy array
        image = npimg.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert it into a colour cv2 image
        # load the neural net.  Should be local to this method as its multi-threaded endpoint
        nets = load_model(CFG, Weights)
        object = do_prediction(image, nets, Lables)
        print(object)
        table = dynamodb.Table('tags')
        params = table.scan()
        
        for index in object:
            for entry in params.get('Items'):
                if ('tags') in entry.keys() and index in entry.get('tags'):
                    if entry.get('id') not in id:
                        id.append(entry.get('id'))
        response_dict['body']=id;
        # print(type(id))
        
        return {
        "statusCode": 200,
        "isBase64Encoded": True,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin":"*"
        },
        "body": json.dumps(id)
        }
        
        # response_dict['statusCode'] =200;
        # return {
        # 'statusCode': 200,
        
        # # 'body': json.dumps(id)
        # # 'body': json.dumps(response_dict)
        #  }

        # return json.make_response(response_dict);
    except Exception as e:

        print("Exception  {}".format(e))
        traceback.print_exc()

# if __name__ == '__main__':
#     main()



