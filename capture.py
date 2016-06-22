import pygame.camera
import time

for i in range (0,3):
    pygame.camera.init()
    pygame.camera.list_cameras()
    cam = pygame.camera.Camera("/dev/video0", (1920, 1080))
    cam.start()
    img = cam.get_image()
    pygame.image.save(img, "pygame" + str(i) + ".jpg")
    cam.stop()
    time.sleep(1)
    print(img)