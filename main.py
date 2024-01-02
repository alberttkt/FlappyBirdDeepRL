import time
import pygame
import gymnasium
from game_state import GameState

def main():
    running_all = True



    game_state = GameState((1280, 720))

    

    previous_key_state = False

    while running_all:
        game_state.restart()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    running_all = False

            # Get the keys only on the rising edge
            current_key_state = pygame.key.get_pressed().count(True) > 0

            # Check if the key is pressed on the rising edge
            some_key_pressed = current_key_state and not previous_key_state

            if game_state.update(some_key_pressed):
                running = False


            game_state.draw()
            # Update the previous key state for the next iteration
            previous_key_state = current_key_state

            # Limit FPS to 60
            # dt is delta time in seconds since the last frame, used for framerate-independent physics

        #death animation
        game_state.handle_death()

        #game over screen


        #wait for key press
        while True:
            a = True
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_all = False
                    break
                if event.type == pygame.KEYDOWN:
                    a = False
                    break
            if not a or not running_all:
                break





if __name__ == '__main__':
    main()
