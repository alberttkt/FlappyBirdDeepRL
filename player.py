import pygame

side_shift = 15
up_shift = 15
gravity = 0.95
friction = 0.95
images_player = ["assets/player.png", "assets/player2.png"]


class Player:
    def __init__(self, position=(0, 0), size=(68, 50)):

        self.size = size

        self.images = [pygame.transform.scale(pygame.image.load(image_path), (self.size[0], self.size[1])) for
                       image_path in images_player]
        self.image_index = 0
        self.current_image_use_count = 0
        self.position = position
        self.speed = 0

    def move(self, is_key_pressed: bool) -> bool:

        # if some key is pressed, change speed
        if is_key_pressed:
            self.speed = - up_shift

        self.speed = self.speed + gravity
        # apply speed to position
        self.position = (self.position[0], self.position[1] + self.speed)

        # apply friction
        self.speed = self.speed * friction

        # stop if speed is too low

        if abs(self.speed) < 0.1:
            self.speed = 0

        # stop if position is too low
        if self.position[1] > 690 - self.size[1]:
            self.position = (self.position[0], 690 - self.size[1])
            self.speed = 0
            return True

        if self.position[1] < 0:
            self.position = (self.position[0], 0)
            self.speed = 0

    def draw(self, screen):

        screen.blit(self.images[self.image_index % len(self.images)],
                    (self.position[0] - self.size[0] / 2, self.position[1] - self.size[1] / 2))
        self.current_image_use_count += 1
        if self.current_image_use_count == 5:
            self.image_index = (self.image_index + 1) % 2
            self.current_image_use_count = 0
        # pygame.draw.rect(screen, rect=(
        #     self.position[0] - self.size[0] / 2, self.position[1] - self.size[1] / 2, self.size[0], self.size[1]),
        #                  color=(255, 0, 0), width=1)

    def get_position(self):
        return self.position

    def restart(self, position=(0, 0)):
        self.images = [pygame.transform.scale(pygame.image.load(image_path), (self.size[0], self.size[1])) for
                       image_path in images_player]
        self.position = position
        self.speed = 0

    def get_size(self):
        return self.size
