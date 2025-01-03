import io
import logging
import socketserver
from http import server
from threading import Condition, Thread
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from gpiozero import DigitalOutputDevice, PWMOutputDevice
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

# GPIO 핀을 사용하여 모터 제어를 위한 장치 초기화
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

# 전진
def motor_go(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed

# 후진
def motor_back(speed):
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed

# 좌회전
def motor_left(speed):
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = 0.0
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed

# 우회전
def motor_right(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = 0.0

# 정지
def motor_stop():
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = 0.0
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = 0.0

#속도 설정
speedSet = 0.4

# 파이카메라 초기화 및 설정
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))

# 스트리밍 출력 클래스
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    # 버퍼 사용될때마다 알림(?)
    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

# 스트리밍 출력 생성 MJPEG 시작
output = StreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(output))

# HTTP 요청을 처리하는 클래스
class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

# 멀티 스레딩을 지원하는 HTTP 서버 클래스
class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

PAGE = """\
<html>
<head>
<title>picamera2 demo</title>
<style>
    img{
        transform: rotate(180deg);
        transform-origin: center center;
    }
</style>
</head>
<body>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""

# 이미지 전처리 
def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200,66))
    image = cv2.GaussianBlur(image,(5,5),0)
    _,image = cv2.threshold(image,100,255,cv2.THRESH_BINARY_INV)
    image = image / 255
    return image

# 객체 인식, 차선 추척
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

    def id_class_name(class_id, classes):
        return classes.get(class_id, "unknown")

    # 모델 로드
    model = cv2.dnn.readNetFromTensorflow('/home/pi/AI_CAR/OpencvDnn/models/frozen_inference_graph.pb',
                                          '/home/pi/AI_CAR/OpencvDnn/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

    # 라인 트레이싱 모델 로드
    lane_model = load_model('/home/pi/AI_CAR/model/lane_navigation_final.h5')
    
    carState = "stop"

    # 사진 프레임 올때 대기 시키고 전 프레임 처리 후 가져오기 
    while True:
        with output.condition:
            output.condition.wait()
            frame_data = output.frame

        # 프레임 위아래 뒤집기
        image = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.flip(image, -1)
        image_height, image_width, _ = image.shape

        # 객체 감지
        model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
        detections = model.forward()

        stop_motors = False

        #인식 객체 표시 및 신호등 감지시 분리하여 색 판별
        for detection in detections[0, 0, :, :]:
            confidence = detection[2]
            if confidence > .5:
                class_id = int(detection[1])
                class_name = id_class_name(class_id, classNames)
                if class_name == 'traffic light':
                    box_x = int(detection[3] * image_width)
                    box_y = int(detection[4] * image_height)
                    box_width = int(detection[5] * image_width)
                    box_height = int(detection[6] * image_height)

                    # 신호등 삼등분 자르기
                    traffic_light_roi = image[box_y:box_height, box_x:box_width]
                    hsv_roi = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)

                    # 각 신호등 범위 설정
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
                    section_height = height_roi // 3

                    # 높이 3등분 색상 구역설정
                    red_section = mask_red[0:section_height, :]
                    yellow_section = mask_yellow[section_height:2*section_height, :]
                    green_section = mask_green[2*section_height:3*section_height, :]

                    red_pixels = cv2.countNonZero(red_section)
                    yellow_pixels = cv2.countNonZero(yellow_section)
                    green_pixels = cv2.countNonZero(green_section)

                    if red_pixels > yellow_pixels and red_pixels > green_pixels:
                        traffic_light_color = "red"
                        motor_stop()
                    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
                        traffic_light_color = "yellow"
                    elif green_pixels > red_pixels and green_pixels > yellow_pixels:
                        traffic_light_color = "green"
                        motor_go(speedSet)
                    else:
                        traffic_light_color = "unknown"
                    
                    print(f"Traffic light color: {traffic_light_color}")

                    cv2.rectangle(image, (box_x, box_y), (box_width, box_height), (23, 230, 210), thickness=1)
                    cv2.putText(image, f"Traffic Light: {traffic_light_color}", (box_x, box_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif class_name == 'person':
                    stop_motors = True
                    print("Person detected: stopping motors")
                else:
                    box_x = int(detection[3] * image_width)
                    box_y = int(detection[4] * image_height)
                    box_width = int(detection[5] * image_width)
                    box_height = int(detection[6] * image_height)
                    cv2.rectangle(image, (box_x, box_y), (box_width, box_height), (23, 230, 210), thickness=1)
                    cv2.putText(image, class_name, (box_x, box_y + int(0.05 * image_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if stop_motors:
            motor_stop()

        # 라인 트레이싱
        preprocessed = img_preprocess(image)
        X = np.asarray([preprocessed])
        steering_angle = int(lane_model.predict(X)[0])
        print("Steering angle:", steering_angle)

        #각도에 따라서 방향 결정
        if steering_angle >= 70 and steering_angle <= 110:
            if carState == "go":
                print("go")
                motor_go(speedSet)
        elif steering_angle > 111:
            if carState == "go":
                print("right")
                motor_right(speedSet)
        elif steering_angle < 71:
            if carState == "go":
                print("left")
                motor_left(speedSet)
        elif carState == "stop":
            motor_stop()

        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            break

# 스트리밍 서버 시작
server = StreamingServer(('0.0.0.0', 8000), StreamingHandler)
server_thread = Thread(target=server.serve_forever)
server_thread.start()

process_frames()

# 자원 해제
picam2.stop_recording()
cv2.destroyAllWindows()
PWMA.value = 0.0
PWMB.value = 0.0
