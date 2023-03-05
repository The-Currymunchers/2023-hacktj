from kivy.app import App
from kivy.uix.camera import Camera


class LiveVideoApp(App):
    def build(self):
        # create a camera object
        camera = Camera()
        # set the camera resolution to the maximum available
        camera.resolution = (640, 480)
        # start the camera
        camera.play()
        return camera


if __name__ == '__main__':
    LiveVideoApp().run()