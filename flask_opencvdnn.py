from flask import Flask, request, render_template, Response
import cv2
import io
import numpy as np
from threading import Condition, Thread

app = Flask(__name__)

camera = cv2.VideoCapture(0)

# StreamingOutput 클래스 정의
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

output = StreamingOutput()

# 객체 인식 및 차선 추적
def process_frames():
    classNames = {0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                  7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 
                  15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 
                  24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 
                  35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 
                  42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 
                  51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 
                  59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 
                  70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 
                  79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 
                  88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

    # 모델 로드
    model = cv2.dnn.readNetFromTensorflow('./frozen_inference_graph.pb',
                                          './ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

    while True:
        with output.condition:
            output.condition.wait()
            frame_data = output.frame
        
        image = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.flip(image, -1)
        image_height, image_width, _ = image.shape

        # 객체 감지
        model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
        detections = model.forward()

        for detection in detections[0, 0, :, :]:
            confidence = detection[2]
            class_id = int(detection[1])
            if confidence > 0.5:
                box_x = int(detection[3] * image_width)
                box_y = int(detection[4] * image_height)
                box_width = int(detection[5] * image_width)
                box_height = int(detection[6] * image_height)

                # 객체 박스 그리기
                cv2.rectangle(image, (box_x, box_y), (box_width, box_height), (23, 230, 210), thickness=1)

                # 'person' (class_id == 1) 또는 'traffic light' (class_id == 10) 감지 시 메시지 출력
                if class_id == 1 or class_id == 10:
                    print("stop")

        # 이미지 인코딩
        _, jpeg = cv2.imencode('.jpg', image)
        output.write(jpeg.tobytes())

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    file.save('received_image.jpg')
    return 'File received'

# 비디오 스트리밍을 위한 경로
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = cv2.imread('received_image.jpg')
            if frame is None:
                # 프레임을 읽지 못했을 경우 대기
                continue
            
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

# 메인 함수
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
