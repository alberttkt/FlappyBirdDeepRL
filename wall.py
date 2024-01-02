import pygame
import random

height = 500
shift = 5
default_hole_size = 210


class Wall:
    windows_size = ()
    number_of_walls = 0

    @classmethod
    def set_window_size(cls, size):
        cls.windows_size = size

    def __init__(self, img_path="assets/pipe.png",width = 80):
        Wall.number_of_walls += 1
        hole_size = random.randint(default_hole_size - min(60,40+ Wall.number_of_walls), default_hole_size + max(-40,60- Wall.number_of_walls))
        self.position = Wall.windows_size[0]
        self.img = pygame.image.load(img_path)
        self.img = pygame.transform.scale(self.img, (width, height))
        self.width = width
        self.hole_start = random.randint(70, Wall.windows_size[1] - hole_size - 70)
        self.hole_end = self.hole_start + hole_size

    def update(self, game_speed, player_position) -> bool:
        old_position = self.position
        self.position = self.position - shift * game_speed
        return old_position > player_position[0] >= self.position

    def draw(self, screen):
        top = pygame.transform.flip(self.img, False, True)
        bottom = self.img
        bottom = bottom.subsurface((0, 0, self.width, self.__class__.windows_size[1] - self.hole_end))
        screen.blit(top, (self.position, -height + self.hole_start))
        screen.blit(bottom, (self.position, self.hole_end))

    def get_position(self):
        return self.position

    def collide(self, position: tuple[float, float], size: (float, float)) -> bool:
        if position[0] + size[0] / 2 < self.position or position[0] - size[0] / 2 > self.position + self.width:
            return False

        if position[1] - size[1] / 2 < self.hole_start or position[1] + size[1] / 2 > self.hole_end:
            return True

        return False

    def in_wall(self, position, player_width) -> bool:
        # start at position-width/2 and ends at position+width/2

        return self.position <= position <= self.position + self.width
