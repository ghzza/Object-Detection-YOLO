import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg') #load the file of YOLOv3 Weights and YOLOv3 Configuration

classes = [] #will be classes of coco
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

#load camera
cap = cv2.VideoCapture('test.mp4') #0 for detecting image using webcam camera



while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB = True, crop = False)

    #TO VIEW BLOB RESULT
    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)
    #         cv2.waitKey(0)

    net.setInput(blob) #to set the input from the blob to the network

    output_layers_names = net.getUnconnectedOutLayersNames() #getUnconnectedOutLayersNames() for extracting the output
    layerOutputs = net.forward(output_layers_names) #run the forward pass and obtain the output, called as layerOutputs

    #to get the result, make a list 3 list
    boxes = [] #extract the bounding boxes
    confidences = [] #store the confidence
    class_ids = [] # predicted classes

    #tuple loops
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:] #create an array that is 'scores' which used to store all the 80 classes predictions
            class_id = np.argmax(scores) #find out the location that contains the higher scores
            confidence = scores[class_id] #assign the result of class_ids to the confidence
            if confidence > 0.5:
                center_x = int(detection[0]*width) #x coordinate of image, multiply with width of image because detection[0] has been normalized
                center_y = int(detection[1]*height) #y coordinate of image, multiply with height of image because detection[1] has been normalized
                w = int(detection[2]*width) #width, multiply with width of image because detection[2] has been normalized
                h = int(detection[3]*height) #height, multiply with height of image because detection[3] has been normalized

                #to get the upper left corner
                x = int(center_x - w/2) 
                y = int(center_y - h/2) 

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    print(len(boxes))
    Indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0,1) #(bboxes, scores, score_threshold, long maximum suppression)
    #print(Indexes.flatten())

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255, size=(len(boxes), 3)) 

    for i in Indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,0,0), 2)

    cv2.imshow('YOLOv3 Object Detection Result', img)
    key = cv2.waitKey(1)
    if key==27:
        break
