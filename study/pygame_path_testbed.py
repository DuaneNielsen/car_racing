import pygame, sys
from pygame import Vector2
from polygon import adjoint_matrix, apply_transform, Polygon, se2_from_adjoint, Circle, PygameCamera
import torch
import pygame.locals
from math import cos, sin, degrees
from env.car_path import Path

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (125, 125, 125)

# clockwise winding required for the clipping to work
H, W = 20., 5.
car = Polygon([
    [W, W, -W, -W],
    [H, -H, -H, H],
], N=2, color=RED)

path = Path(n_cars=2)
path.attach(car)


def main():
    pygame.init()
    clock = pygame.time.Clock()

    DISPLAY = pygame.display.set_mode((1000, 800), 0, 32)
    camera = PygameCamera(DISPLAY,
                          se2=torch.tensor([500., 500., 0]),
                          scale=torch.ones(2)*3.5, resolution=(1000, 800))

    DISPLAY.fill(WHITE)
    t = 0.0

    car.pos = torch.tensor([[-20, 0], [20, 0]])

    while True:
        clock.tick(25)
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()

        # draw the red rotating rectangle on the blue clipping rectangle
        DISPLAY.fill(WHITE)
        theta = torch.tensor([t, t-0.2])
        length = torch.tensor([t*100., t*50.+0.2])

        path.set_path_params(theta, length)
        camera.draw(car)
        path.draw(camera)

        t += 0.001
        camera.draw_text([f'{degrees(t):.2f}'], GREEN)

        pygame.display.flip()


main()
