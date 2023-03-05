from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.config import Config
from kivy.graphics.texture import Texture
import numpy as np
from cvzone.ColorModule import ColorFinder
import cv2

Config.set('graphics', 'height', '360')
Config.set('graphics', 'width', '640')

class CamApp(App):

    def build(self):
        # Create two BoxLayout widgets
        top_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.9))
        bottom_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))

        # Add Image widget to top_layout
        self.img1 = Image()
        top_layout.add_widget(self.img1)

        # Welcome button
        welcome_button = Button(text="Welcome To Sports Ball Detection!", size_hint=(1, 0.1))
        welcome_button.bind(on_press=self.hide_button)
        top_layout.add_widget(welcome_button)
        
        # Start Stream button
        start_button = Button(text="Start Stream", size_hint=(0.5, 1), background_color=[0, 0, 1, 1])
        start_button.bind(on_press=self.start_stream)
        bottom_layout.add_widget(start_button)
        
        # End Stream button
        end_button = Button(text="End Stream", size_hint=(0.5, 1), background_color=[1, 0, 0, 1])
        end_button.bind(on_press=self.end_stream)
        bottom_layout.add_widget(end_button)
        
        # Add both layouts to a parent BoxLayout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(top_layout)
        layout.add_widget(bottom_layout)

        # opencv2 stuffs
        self.capture = None
        self.ballColorFinder = ColorFinder(False)
        self.hsvVals = {'hmin': 11, 'smin': 190, 'vmin': 131, 'hmax': 26, 'smax': 255, 'vmax': 255}
        self.stream_active = False

        self.welcome_button = welcome_button  # Save reference to welcome button

        return layout

    def hide_button(self, instance):
        # Hide the welcome button when pressed
        self.welcome_button.visible = False

    def start_stream(self, instance):
        if not self.stream_active:
            self.capture = cv2.VideoCapture(0)
            Clock.schedule_interval(self.update, 1.0/33.0)
            self.stream_active = True

    def end_stream(self, instance):
        if self.stream_active:
            self.capture.release()
            self.stream_active = False

    def update(self, dt):
        if self.capture is None:
            return

        # display image from cam in opencv window
        ret, frame = self.capture.read()
        black = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        circles = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        imageColor, mask = self.ballColorFinder.update(frame, self.hsvVals)

        blurFrame = cv2.GaussianBlur(mask, (17, 17), 0)
        contours, _ = cv2.findContours(blurFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(black, contours, -1, (0,255,0), 3)

        if contours:
            c = max(contours, key=cv2.contourArea)

            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(c)
        
            # Draw a green circle around the pingpong ball
            cv2.circle(frame, (x + w//2, y + h//2), w//2, (0,255,0), -1)
            cv2.circle(circles, (x + w//2, y + h//2), w//2, (0,255,0), -1)


        # cv2.imshow("CV2 Image", frame)
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = texture1

if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()