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
import uuid
from urllib.parse import unquote_plus

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
    # objects = []
    detection = []
    # detection['rectangle'] = {}
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            if (confidences[i]  > 0.5):
                if (format(LABELS[classIDs[i]])) not in detection:
                    detection.append(format(LABELS[classIDs[i]]))
                
    return (detection)  # returns  an empty list of no parameters are recognised




yolo_path  = os.getcwd()

## Yolov3-tiny versrion

s3_client = boto3.client('s3')
dynamodb = boto3.client('dynamodb')
TABLE_NAME = 'tags'

labelsPath = s3_client.get_object(Bucket = 'tagtag-yolo-config', Key = 'coco.names')
cfgpath = s3_client.get_object(Bucket='tagtag-yolo-config', Key = 'yolov3-tiny.cfg')
wpath = s3_client.get_object(Bucket='tagtag-yolo-config', Key = 'yolov3-tiny.weights')

Lables = labelsPath['Body'].read().decode('utf8').strip().split("\n")
CFG = cfgpath['Body'].read()
Weights = wpath['Body'].read()

def lambda_handler(event, context):
    try:
        response_dict = {}
        
        
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = unquote_plus(record['s3']['object']['key'])
            print("File {0} uploaded to {1} bucket".format(key, bucket))
            # location = boto3.client('s3').get_bucket_location(Bucket=bucket)['LocationConstraint']
            imagefile = s3_client.get_object(Bucket=bucket, Key=key)
            # url = "https://s3-%s.amazonaws.com/%s/%s" % (location, bucket, key)
            url = f'https://{bucket}.s3.amazonaws.com/{key}'
            print(url)
        
        # imagefile = base64.b64decode(event['image'].encode('utf-8'))  # decode the encoded image sent by the client
        
    
            npimg = np.frombuffer(imagefile['Body'].read(), np.uint8)  # Interpret a buffer as a 1-dimensional array
            img = cv2.imdecode(npimg,
                               -1)  # The function imdecode reads an image from the specified buffer in the memory.
            npimg = np.array(img)  # convert the read image to a numpy array
            image = npimg.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert it into a colour cv2 image
            # load the neural net.  Should be local to this method as its multi-threaded endpoint
            nets = load_model(CFG, Weights)
            tags = do_prediction(image, nets, Lables)
        

        # convert the prediction into the desired format in the Assignment
        if len(tags)>0:
            response_dict['id'] = {'S':str(url)}  # 
            response_dict['tags'] = {'SS':tags}
            response = dynamodb.put_item(TableName=TABLE_NAME, Item=response_dict)
        else:
            response_dict['id'] = {'S':str(url)}  # 
            #response_dict['tags'] = {'SS': [] }
            response = dynamodb.put_item(TableName=TABLE_NAME, Item=response_dict)
        
          
        print(response_dict)
        
        return {
        'statusCode': 200,
        'body': json.dumps('Records successfully inserted into database...')
        }
                
        

    except Exception as e:

        print("Exception  {}".format(e))

# if __name__ == '__main__':
#     main()