from flask import Flask, request, render_template, Response, jsonify
import cv2
import io
import numpy as np
from threading import Condition, Thread
import requests  # HTTP 요청을 위해 추가

#과부화 방지 임포트
import time
import json

# Flask 앱 초기화
app = Flask(__name__)

# 웹캠 초기화
camera = cv2.VideoCapture(0)

# StreamingOutput 클래스 : 비디오 스트리밍을 위한 클래스 정의
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
    print("process_frames 스레드가 시작되었습니다.")  # 스레드 시작 확인

    # COCO 데이터셋 클래스 정의 (객체 인식용)
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

    # SSD MobileNet 모델 로드
    model = cv2.dnn.readNetFromTensorflow('./frozen_inference_graph.pb',
                                          './ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

    # 프레임 처리 로직
    while True:
        # 1. 객체 감지
        # 2. 신호등 색상 인식
        # 3. 라즈베리파이로 제어 명령 전송
        
        with output.condition:
            output.condition.wait()
            frame_data = output.frame
        
        if frame_data is None:
            print("frame_data가 None입니다.")
            continue

        # pc과부화 방지 위한 코드
        start_time = time.time() #시작 시간을 기록
        
        image = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.flip(image, -1)
        image_height, image_width, _ = image.shape

        # 1. 객체 감지
        model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
        detections = model.forward()

        for detection in detections[0, 0, :, :]:
            confidence = detection[2]
            class_id = int(detection[1])
            if confidence > 0.6:
                box_x = int(detection[3] * image_width)
                box_y = int(detection[4] * image_height)
                box_width = int(detection[5] * image_width)
                box_height = int(detection[6] * image_height)

                # 객체 박스 그리기
                cv2.rectangle(image, (box_x, box_y), (box_width, box_height), (23, 230, 210), thickness=1)

                # 'person' (class_id == 1) 또는 'traffic light' (class_id == 10) 감지 시 메시지 출력
                if class_id == 1:
                    print("human detected!")
                    send_message_to_raspberry_pi("stop")
                if class_id == 10:
                    print("traffic light detected!")
                    #send_message_to_raspberry_pi("Traffic light detected!")

                    # 2. 신호등 색상 인식
                    if box_x < 0 or box_y < 0 or box_width <= box_x or box_height <= box_y:
                        continue

                    traffic_light_roi = image[box_y:box_height, box_x:box_width]

                    if traffic_light_roi.size == 0:
                        print("skip")
                        continue

                    hsv_roi = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)

                    lower_red1 = np.array([0, 70, 50])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([170, 70, 50])
                    upper_red2 = np.array([180, 255, 255])
                    lower_yellow = np.array([15, 70, 50])
                    upper_yellow = np.array([35, 255, 255])
                    lower_green = np.array([40, 70, 50])
                    upper_green = np.array([90, 255, 255])

                    mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
                    mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
                    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                    mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
                    mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)

                    
                    height_roi, width_roi, _ = traffic_light_roi.shape
                    half_height = height_roi // 2

                    green_section = mask_red[0:half_height, :]
                    red_section = mask_green[half_height:height_roi, :]

                    red_pixels = cv2.countNonZero(red_section)
                    green_pixels = cv2.countNonZero(green_section)

                    # 3. 라즈베리파이로 제어 명령 전송
                    if red_pixels > green_pixels:
                        traffic_light_color = "red"
                        send_message_to_raspberry_pi("stop")
                        print("Red light detected. Motor should stop.")
                    else:
                        traffic_light_color = "green"
                        send_message_to_raspberry_pi("auto")
                        print("Green light detected. Motor should go.")
                    
                    #테스트 코드
                    red_pixels = cv2.countNonZero(mask_red)
                    yellow_pixels = cv2.countNonZero(mask_yellow)
                    green_pixels = cv2.countNonZero(mask_green)

                    # 색상 판별
                    if red_pixels > max(green_pixels, yellow_pixels):
                        send_message_to_raspberry_pi("stop")
                        print("Red light detected. Motor should stop.")
                        traffic_light_color = "red"
                    elif green_pixels > max(red_pixels, yellow_pixels):
                        send_message_to_raspberry_pi("go")
                        print("Green light detected. Motor should go.")
                        traffic_light_color = "green"
                    elif yellow_pixels > max(red_pixels, green_pixels):
                        print("Yellow light detected. Caution advised.")
                        traffic_light_color = "yellow"

                    print(f"Traffic light color: {traffic_light_color}")

        # 이미지 인코딩
        _, jpeg = cv2.imencode('.jpg', image) 
        output.write(jpeg.tobytes())

        # 처리 시간에 따라 딜레이를 추가하여 과부화 방지 
        elapsed_time = time.time() - start_time
        if elapsed_time < 0.01:
            time.sleep(0.01 - elapsed_time)

# 라즈베리파이로 메시지 전송하는 함수
def send_message_to_raspberry_pi(message):
    # url = 'http://192.168.137.36:5000/receive_message'  # 라즈베리파이의 IP와 포트로 설정
    url = 'http://192.168.137.34:5000/receive_message'  # 라즈베리파이의 IP와 포트로 설정
    try:
        response = requests.post(url, json={"message": message})
        print(f"Response from Raspberry Pi: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send message to Raspberry Pi: {e}")


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
                continue
            
            _, jpeg = cv2.imencode('.jpg', frame)
            output.write(jpeg.tobytes())
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

# 메인 함수
if __name__ == '__main__':
    process_thread = Thread(target=process_frames)
    process_thread.daemon = True
    process_thread.start()
    print("Flask 서버가 시작됩니다.")
    
    app.run(host='0.0.0.0', port=5000)
