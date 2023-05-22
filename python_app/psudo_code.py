
import cv2
import os
import numpy as np

root = os.path.dirname(__file__)
model_path = os.path.join(root, 'models/yolo_onnx')


def detect(video_path, confidence, nms_threshold):
    # read labels
    labels = open(os.path.join(model_path, "labels.txt")).read().strip().split("\n")

    # load trained object detector model in onnx format
    net = cv2.dnn.readNetFromONNX(os.path.join(model_path, "best.onnx"))

    # initialize the video stream
    vs = cv2.VideoCapture(video_path)
    (W, H) = (None, None)
    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # construct a blob from the input frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        # perform a forward pass of the YOLO object detector,
        net.setInput(blob)
        layerOutputs = net.forward()

        # initialize our lists of detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                scores = detection[5:]
                classID = np.argmax(scores)
                conf = scores[classID]

                # filter out weak predictions by given threshold
                if conf > confidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(conf))
                    classIDs.append(classID)
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms_threshold)
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), "red", 2)
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, "red", 2)

        cv2.imshow("Video Output", frame.astype('uint8'))
        # Press Q on keyboard to abort
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    vs.release()
    exit(1)
