import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class CameraApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        while True:
            ret, frame = self.capture.read()
            if ret:
                # Run your OpenCV pipeline here
                # You can access the camera stream using the "frame" variable
                pass

if __name__ == '__main__':
    CameraApp().run()