import cv2
import numpy as np
import time

#load Yolo
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(255, 0, size=(len(classes), 3))

#loading image
cap = cv2.VideoCapture(0)


starting_time = time.time()
frame_id = 0

font = cv2.FONT_HERSHEY_PLAIN
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    #detecting objets 
    #blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    #show information on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        #extract only lines with scores greater than confidence 0.5
        out = out[np.where(out[:, 5:] > 0.5)[0],:]
        for detection in out:
            scores = detection[5: ] 
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: #era 0.5
                #object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1]* height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), 2)
                #Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)

    

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 12), font, 0.8, color, 1)
            

    #calculando frame por segundo
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time

    #imprimindo o fps a tela 
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 40), font, 3, (0, 0, 0), 2)


    cv2.imshow("Image", frame)

    #cv2.imwrite("Cadeira YOLO.jpg", frame)

    key = cv2.waitKey(1)
    
    if key == 27:
       break
    
cap.release()
cv2.destroyAllWindows()