import time
import mycamera
import cv2
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice

PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

def motor_go(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed

def motor_back(speed):
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed
    
def motor_left(speed):
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = 0.0
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed
    
def motor_right(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = 0.0

def motor_stop():
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = 0.0
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = 0.0

speedSet = 0.5

camera = mycamera.MyPiCamera(640,480)
        
def main():
    filepath = "/home/pi/AI_CAR/video/train"
    i = 0
    carState = "stop"
    try:
        while True:
            
            keyValue = cv2.waitKey(1)
        
            if keyValue == ord('q'):
                break
            elif keyValue == 82:
                print("go")
                carState = "go"
                motor_go(speedSet)
            elif keyValue == 84:
                print("stop")
                carState = "stop"
                motor_stop()
            elif keyValue == 81:
                print("left")
                carState = "left"
                motor_left(speedSet)
            elif keyValue == 83:
                print("right")
                carState = "right"
                motor_right(speedSet)
                
            _, image = camera.read()
            image = cv2.flip(image,-1)
            height, _, _ = image.shape
            image = image[int(height/2):,:,:]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            image = cv2.resize(image, (200,66))
            image = cv2.GaussianBlur(image,(5,5),0)
            _,image = cv2.threshold(image,200,255,cv2.THRESH_BINARY_INV)
            
            if carState == "left":
                cv2.imwrite("%s_%05d_%03d.png" % (filepath, i, 45), image)
                i += 1
            elif carState == "right":
                cv2.imwrite("%s_%05d_%03d.png" % (filepath, i, 135), image)
                i += 1
            elif carState == "go":
                cv2.imwrite("%s_%05d_%03d.png" % (filepath, i, 90), image)
                i += 1
            
            cv2.imshow('Original', image)
            
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
    PWMA.value = 0.0
    PWMB.value = 0.0