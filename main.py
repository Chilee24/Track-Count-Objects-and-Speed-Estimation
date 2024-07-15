import cv2
import numpy as np
import math
from ultralytics import YOLO

START_POINT = 80
END_POINT = 150
CLASSES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush"]
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

CONFIDENCE_SETTING = 0.4
MAX_DISTANCE = 80

def get_output_layers(net):
    try:
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    except:
        print("Can't get output layers")
        return None

def detections_yolov8(model, image, confidence_setting, frame_w, frame_h, classes=None):
    results = model(image)[0]
    boxes = []
    class_ids = []
    confidences = []

    for result in results.boxes:
        class_id = int(result.cls)
        confidence = float(result.conf)
        if confidence > confidence_setting and class_id in VEHICLE_CLASSES:
            print("Object name: " + classes[class_id] + " - Confidence: {:0.2f}".format(confidence * 100))
            box = result.xyxy[0]
            x, y, x2, y2 = map(int, box)
            w = x2 - x
            h = y2 - y
            class_ids.append(class_id)
            confidences.append(confidence)
            boxes.append([x, y, w, h])
    return boxes, class_ids, confidences

def draw_prediction(classes, colors, img, class_id, confidence, x, y, width, height):
    try:
        label = str(classes[class_id])
        color = colors[class_id]
        center_x = int(x + width / 2.0)
        center_y = int(y + height / 2.0)
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)

        cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)
        cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)
        cv2.putText(img, label + ": {:0.2f}%".format(confidence * 100), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    except (Exception, cv2.error) as e:
        print("Can't draw prediction for class_id {}: {}".format(class_id, e))

def check_location(box_y, box_height, height):
    center_y = int(box_y + box_height / 2.0)
    return center_y > height - END_POINT

def check_start_line(box_y, box_height):
    center_y = int(box_y + box_height / 2.0)
    return center_y > START_POINT

def counting_vehicle_yolov8(video_input, video_output, skip_frame=1):
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    model = YOLO('yolov8n.pt')  # Use YOLOv8 nano model

    cap = cv2.VideoCapture(video_input)
    ret_val, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]

    video_format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output, video_format, 25, (width, height))

    list_object = []
    number_frame = 0
    number_vehicle = 0
    while cap.isOpened():
        number_frame += 1
        ret_val, frame = cap.read()
        if frame is None:
            break

        tmp_list_object = list_object
        list_object = []
        for obj in tmp_list_object:
            tracker = obj['tracker']
            class_id = obj['id']
            confidence = obj['confidence']
            check, box = tracker.update(frame)
            if check:
                box_x, box_y, box_width, box_height = box
                draw_prediction(CLASSES, colors, frame, class_id, confidence,
                                box_x, box_y, box_width, box_height)
                obj['tracker'] = tracker
                obj['box'] = box
                if check_location(box_y, box_height, height):
                    number_vehicle += 1
                else:
                    list_object.append(obj)

        if number_frame % skip_frame == 0:
            boxes, class_ids, confidences = detections_yolov8(model, frame, CONFIDENCE_SETTING,
                                                              width, height, classes=CLASSES)
            for idx, box in enumerate(boxes):
                box_x, box_y, box_width, box_height = box
                if not check_location(box_y, box_height, height):
                    box_center_x = int(box_x + box_width / 2.0)
                    box_center_y = int(box_y + box_height / 2.0)
                    check_new_object = True
                    for tracker in list_object:
                        current_box_x, current_box_y, current_box_width, current_box_height = tracker['box']
                        current_box_center_x = int(current_box_x + current_box_width / 2.0)
                        current_box_center_y = int(current_box_y + current_box_height / 2.0)
                        distance = math.sqrt((box_center_x - current_box_center_x) ** 2 +
                                             (box_center_y - current_box_center_y) ** 2)
                        if distance < MAX_DISTANCE:
                            check_new_object = False
                            break
                    if check_new_object and check_start_line(box_y, box_height):
                        new_tracker = cv2.TrackerKCF_create()
                        new_tracker.init(frame, tuple(box))
                        new_object = {
                            'id': class_ids[idx],
                            'tracker': new_tracker,
                            'confidence': confidences[idx],
                            'box': box
                        }
                        list_object.append(new_object)
                        draw_prediction(CLASSES, colors, frame, new_object['id'], new_object['confidence'],
                                        box_x, box_y, box_width, box_height)

        cv2.putText(frame, "Number : {:03d}".format(number_vehicle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.line(frame, (0, START_POINT), (width, START_POINT), (204, 90, 208), 1)
        cv2.line(frame, (0, height - END_POINT), (width, height - END_POINT), (255, 0, 0), 2)
        cv2.imshow("Counting", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    counting_vehicle_yolov8('videoTest\\highway.mp4', 'vehicles_yolov8.avi')
