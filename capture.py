import pygame.camera
import time

#Initialize image size on screen
screen = pygame.display.set_mode((1920, 1080))

#Start camera
pygame.camera.init()
pygame.camera.list_cameras()
cam = pygame.camera.Camera("/dev/video0", (1920, 1080))
cam.start()

while (1):
    #Take picture
    img = cam.get_image()

    #Show image on screen
    screen.blit(img,(0,0))
    pygame.display.flip()

    #Save image to file (comment out if you take many images)
    #pygame.image.save(img, "pygame" + str(i) + ".jpg")
    time.sleep(0.1)

#Stop the camera
cam.stop()