import pygame, sys
from pygame.locals import *
import numpy as np
from numpy import pi, sin, cos
from pygame import Vector2
from polygon_old import PolygonClipper, Polygon, polygonArea, from_pygame_poly


# clockwise winding required for the clipping to work
rect = np.array([
    [50., 50., -50., -50.],
    [50., -50., -50., 50.],
    [1., 1., 1., 1.]
])

fixed = Polygon(rect)
fixed.pos = Vector2(200, 100)
fixed.theta = 0.

rotator = Polygon(rect)
rotator.pos = Vector2(200, 100)
rotator.theta = 0.
rotator.scale = Vector2(1.1, 1.1)

clip = PolygonClipper()

road_seg = Polygon(rect)
road_seg.pos = Vector2(400, 100)


def main():
    pygame.init()
    clock = pygame.time.Clock()

    DISPLAY = pygame.display.set_mode((500, 400), 0, 32)

    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    GRAY = (125, 125, 125)

    DISPLAY.fill(WHITE)


    #rect2 = pygame.Rect(200, 350, 100, 20)
    # rect2 = rotate2D(shape2vector(rect2), pi / 4)
    start, end = pygame.Vector2(100, 150), pygame.Vector2(350, 200)
    # start_inside, end_inside = rect.clipline(start, end)

    # pygame.draw.line(DISPLAY, RED, start, start_inside, width=2)
    # pygame.draw.line(DISPLAY, GREEN, start_inside, end_inside, width=2)
    # pygame.draw.line(DISPLAY, RED, end_inside, end, width=2)

    while True:
        clock.tick(25)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # draw the red rotating rectangle on the blue clipping rectangle
        rotator.theta += pi / 40
        DISPLAY.fill(WHITE)
        pygame.draw.polygon(DISPLAY, BLUE, fixed.pygame_world_verts)
        area_rotator = polygonArea(rotator.world)
        clipped_verts = clip(rotator.pygame_world_verts, fixed.pygame_world_verts)
        clipped_area_rotator = polygonArea(from_pygame_poly(clipped_verts))
        assert area_rotator != clipped_area_rotator
        pygame.draw.polygon(DISPLAY, RED, clipped_verts)

        pygame.draw.polygon(DISPLAY, GRAY, road_seg.pygame_world_verts)
        pygame.display.flip()


main()
