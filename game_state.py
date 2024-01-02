import pygame
from numba import jit

from player import Player
from wall import Wall
from walls import Walls
import time

DEFAULT_GAME_SPEED = 1

background_path = "assets/background.png"


class GameState:

    def __init__(self, window_size=(1280, 720)):
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        self.background = pygame.image.load(background_path)
        self.clock = pygame.time.Clock()
        self.background = pygame.transform.scale(self.background, window_size)
        window_size_without_bottom = (window_size[0], window_size[1] - 60)
        self.player = Player(position=(window_size_without_bottom[0] / 2, window_size_without_bottom[1] / 2))
        self.window_size = window_size_without_bottom
        self.walls = Walls(window_size_without_bottom)
        self.game_speed = DEFAULT_GAME_SPEED
        self.score = 0
        self.high_score = 0

    def restart(self):
        self.game_speed = DEFAULT_GAME_SPEED
        self.score = 0
        self.player.restart(position=(self.window_size[0] / 2, self.window_size[1] / 2))
        self.walls.restart()


    def draw(self,screen=None,background=True):
        display = False
        if screen is None:
            display = True
            screen = self.screen
        if background:
            screen.blit(self.background, (0, 0))
        else:
            screen.fill((0,0,0))
        self.player.draw(screen)
        self.walls.draw(screen)
        font = pygame.font.Font('freesansbold.ttf', 32)

        if display:
            text = font.render(f"Score: {self.score}", True, "black")
            screen.blit(text, (20, 20))
            text = font.render(f"High Score: {self.high_score}", True, "black")
            screen.blit(text, (20, 60))
            pygame.display.flip()


    def update(self, is_key_pressed: bool):
        if self.player.move(is_key_pressed):
            return True
        if self.walls.update(self.game_speed, self.player.get_position()):
            self.score += 1
            self.high_score = max(self.high_score, self.score)
            if self.score % 10 == 0:
                self.game_speed += 0.1
        self.clock.tick(60)

        return self.collide()

    def collide(self):
        return self.walls.collide(self.player.get_position(), self.player.get_size())

    def get_score(self):
        return self.score

    def get_final_position(self):
        return self.walls.gravity_destination(self.player.get_position()[0], self.player.size)

    def handle_death(self):
        self.player.images = [pygame.transform.scale(pygame.image.load("assets/dead_player.png"), self.player.size)]
        self.draw()
        pygame.display.flip()
        final_position = self.get_final_position()
        drop = 10
        while self.player.position[1] < final_position:
            self.player.position = (self.player.position[0], min(final_position, self.player.position[1] + drop))
            drop += 2
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        font = pygame.font.Font('freesansbold.ttf', 64)
        text = font.render(f"Game Over", True, "black")
        self.screen.blit(text, (
        self.screen.get_size()[0] / 2 - text.get_size()[0] / 2, self.screen.get_size()[1] / 2 - text.get_size()[1] / 2 - 50))
        text = font.render(f"Press any key to restart", True, "black")
        self.screen.blit(text, (
        self.screen.get_size()[0] / 2 - text.get_size()[0] / 2, self.screen.get_size()[1] / 2 - text.get_size()[1] / 2 + 50))
        pygame.display.flip()

    def get_image(self):
        screen = pygame.Surface(self.window_size)
        self.draw(screen,background=False)

        return screen
