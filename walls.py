import pygame
import random

from wall import Wall



class Walls:
    next_space_between = random.randint(350, 450)

    def __init__(self, window_size):
        Wall.set_window_size(window_size)
        self.window_size = window_size
        self.__walls: list[Wall] = []

    def update(self, game_speed, player_position):
        increase_score = False
        for wall in self.__walls:
            increase_score = wall.update(game_speed, player_position) or increase_score

        if self.should_pop():
            self.__walls.pop(0)

        if self.can_add():
            self.add_wall()

        return increase_score

    def draw(self, screen):
        for wall in self.__walls:
            wall.draw(screen)

    def can_add(self):
        return len(self.__walls) == 0 or self.__walls[-1].get_position() + Walls.next_space_between < self.window_size[0]

    def add_wall(self):
        Walls.next_space_between = random.randint(350, 450)
        self.__walls.append(Wall())

    def collide(self, position: tuple[float, float], size: (float, float)) -> bool:
        for wall in self.__walls:
            if wall.collide(position, size):
                return True
        return False

    def should_pop(self):
        return len(self.__walls) > 0 and self.__walls[0].get_position()+80 < 0

    def restart(self):
        self.__walls = []
        Wall.number_of_walls = 0

    def gravity_destination(self, position, size: (int, int)) -> (bool, int):
        for wall in self.__walls:
            if wall.in_wall(position, size[0]):
                return wall.hole_end - size[1]/2
        return Wall.windows_size[1]- size[1]/2

    def get_observation(self):
        return [{
            "top_left": (wall.hole_start,wall.position),
            "bottom_right": (wall.hole_end,wall.position+wall.width)
        } for wall in self.__walls]

