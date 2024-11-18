# 자율주행 제어 (라즈베리파이)
# 모터 제어 및 자율주행 로직

import threading
import time
import mycamera
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from gpiozero import DigitalOutputDevice, PWMOutputDevice
import requests
from flask import Flask, request

# Flask 앱 초기화
app = Flask(__name__)

# 모터 제어 핀 설정
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

# 모터 제어 핀 설정 및 함수
def motor_go(speed): # 전진 동작 제어
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed

def motor_back(speed): # 후진 동작 제어
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed
    
def motor_left(speed): # 좌회전 동작 제어
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = 0.0
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed
    
def motor_right(speed): # 우회전 동작 제어
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = 0.0

def motor_stop(): # 브레이크 동작 제어
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = 0.0
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = 0.0

speedSet = 0.4

# 이미지 전처리 함수
def img_preprocess(image):
    # 상단 절반 제거 (불필요한 영역 제거)
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]  
    
    # YUV 색공간 변환 (조명 영향 최소화)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) 
    
    # 크기 조정 (모델 입력 크기에 맞춤)
    image = cv2.resize(image, (200,66))
    
    # 노이즈 제거
    image = cv2.GaussianBlur(image,(5,5),0) 
    
    # 이진화 처리
    _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 정규화
    image = image / 255
    return image

camera = mycamera.MyPiCamera(640, 480)

def upload_image(image_path):
    with open(image_path, 'rb') as img:
        # MacBook IP address
        requests.post('http://192.168.137.237:5000/upload', files={'file': img})
        
        #requests.post('http://192.168.45.230:5000/upload', files={'file': img})


# 전역 변수 선언
carState = "stop"   # 현재 자동차 상태
lastCommand = ""    # 마지막으로 받은 명령을 저장
auto_mode = False   # 자동 모드 상태를 관리하는 변수

@app.route('/receive_message', methods=['POST'])
def receive_message():
    global carState
    global lastCommand
    global auto_mode
    data = request.json
    if 'message' in data:
        message = data['message']
        print(f"Received message: {message}")
        
        # 새 명령어가 들어왔을 때만 carState 업데이트
        if message != lastCommand:
            lastCommand = message
            if "stop" in message:
                carState = "stop"
                motor_stop()
                auto_mode = False  # 수동 모드로 전환
            elif "go" in message:
                if not auto_mode:
                    carState = "go"  # 수동 모드에서만 동작
            elif "back" in message:
                if not auto_mode:
                    carState = "back"  # 후진 동작 추가
            elif "left" in message:
                if not auto_mode:
                    carState = "left"
            elif "right" in message:
                if not auto_mode:
                    carState = "right"
            elif "auto" in message:
                carState = "auto"  # 자율주행 모드
                auto_mode = True   # 자동 모드 활성화
    return 'Message received'

# Flask motor control
@app.route('/control', methods=['POST'])
def control_car():
    global carState
    global lastCommand
    global auto_mode
    data = request.get_json()  # 안드로이드에서 보낸 JSON 데이터 수신
    command = request.json.get('command')

    # 명령어가 바뀌었을 때만 carState 업데이트
    if command != lastCommand:
        lastCommand = command
        if command == 'go':
            if not auto_mode:
                carState = "go"
                print("android go (manual mode)")
        elif command == 'back':
            if not auto_mode:
                carState = "back"
                print("android back (manual mode)")  # 후진 명령 로깅
        elif command == 'left':
            if not auto_mode:
                carState = "left"
                print("android left (manual mode)")
        elif command == 'right':
            if not auto_mode:
                carState = "right"
                print("android right (manual mode)")
        elif command == 'auto':
            carState = "auto"
            auto_mode = True  # 자율주행 모드로 전환
            print("android auto mode")
        elif command == 'stop':
            carState = "stop"
            motor_stop()
            auto_mode = False  # 수동 모드로 전환
    return "Car command executed."

# 메인 제어 루프
def main():
    global carState  # 전역 변수 사용
    global auto_mode # 자동 모드 상태 관리

    # 딥러닝 모델 로드
    model_path = '/home/pi/AI_CAR/model/lane_navigation_final1.h5'
    model = load_model(model_path)
    
    try:
        while True:
            keyValue = cv2.waitKey(1)
        
            if keyValue == ord('q'):
                break
            elif keyValue == 82:
                print("go")
                if not auto_mode:
                    carState = "go"
            elif keyValue == 84:
                print("stop")
                carState = "stop"
            elif keyValue == 81:
                print("left")
                if not auto_mode:
                    carState = "left"
            elif keyValue == 83:
                print("right")
                if not auto_mode:
                    carState = "right"
                
            _, image = camera.read()
            image = cv2.flip(image, -1)
            preprocessed = img_preprocess(image)
            cv2.imshow('pre', preprocessed)
            
            # 이미지 저장 후 업로드
            cv2.imwrite('image.jpg', image)
            threading.Thread(target=upload_image, args=('image.jpg',)).start()

            X = np.asarray([preprocessed])
            steering_angle = int(model.predict(X)[0])
            # print("predict angle:", steering_angle)
                
            # 현재 상태에 따른 동작 유지 (자동 모드)
            if auto_mode:
                if 70 <= steering_angle <= 110:
                    print("auto go")
                    motor_go(speedSet)
                elif steering_angle > 111:
                    print("auto right")
                    motor_right(speedSet)
                elif steering_angle < 71:
                    print("auto left")
                    motor_left(speedSet)
            else:  # 수동 모드
                if carState == "go":
                    motor_go(speedSet)
                elif carState == "stop":
                    motor_stop()
                elif carState == "left":
                    motor_left(speedSet)
                elif carState == "right":
                    motor_right(speedSet)
                elif carState == "back":  # 후진 상태에서 motor_back 호출
                    motor_back(speedSet)
            
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    # Flask 서버를 별도의 스레드에서 실행
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000))
    flask_thread.daemon = True
    flask_thread.start()

    # 메인 함수 실행
    main()

    cv2.destroyAllWindows()
    PWMA.value = 0.0
    PWMB.value = 0.0
