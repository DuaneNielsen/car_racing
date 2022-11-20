import pygame, sys
from pygame import Vector2
from polygon import adjoint_matrix, apply_transform, Polygon, se2_from_adjoint, Circle, PygameCamera
import torch
import pygame.locals
from math import cos, sin

# clockwise winding required for the clipping to work
H, W = 20., 5.
car = Polygon([
    [W, W, -W, -W],
    [H, -H, -H, H],
], N=2)

rectm = Polygon([
    [W/2, W/2, -W/2, -W/2],
    [1., 0, 0, 1.],
], N=2)
car.attach(rectm)

circlem = Circle(radius=W*2, N=2)
car.attach(circlem)


def joint_transform(theta, l):
    return adjoint_matrix(torch.tensor([[0., l, theta]]))


def forward_kinematics_dh(theta1, length1, theta2, length2, theta3, length3):
    T01 = joint_transform(theta1, length1)
    T12 = joint_transform(theta2, length2)
    T23 = joint_transform(theta3, length3)
    J1 = T01
    J2 = T01.matmul(T12)
    J3 = T01.matmul(T12.matmul(T23))
    return (J1, J2, J3), (length1, length2, length3)
#

def main():
    pygame.init()
    clock = pygame.time.Clock()

    DISPLAY = pygame.display.set_mode((1000, 800), 0, 32)
    camera = PygameCamera(DISPLAY,
                          se2=torch.tensor([500., 500., 0]),
                          scale=torch.ones(2)*3.5, resolution=(1000, 800))

    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    GRAY = (125, 125, 125)

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

        theta1 = t
        theta2 = t
        theta3 = t

        (J1, J2, J3), (l1, l2, l3) = forward_kinematics_dh(theta1, 0., theta2, H, theta3, H)

        def draw_joint(T, color, length=1.):
            se2 = se2_from_adjoint(T)
            rectm.se2 = se2
            circlem.se2 = se2
            rectm.scale = torch.tensor([[1., length]])
            camera.draw_polygon(rectm.world(), color)
            camera.draw_circle(*circlem.world(), color)

        vec = torch.zeros_like(car.pos)
        vec[:, 0] = sin(t)
        car.pos += vec
        camera.draw_polygon(car.world(), RED)
        draw_joint(J1, BLUE, H)
        draw_joint(J2, GRAY, l2)
        draw_joint(J3, GREEN, l3)

        t += 0.03

        pygame.display.flip()


main()
