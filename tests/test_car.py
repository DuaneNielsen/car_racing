import pygame
from env.car_path import Car, CarRacingPathEnv
from polygon import PygameCamera, Polygon, adjoint_matrix, apply_transform
import torch
import sys

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (125, 125, 125)


class PygameManager:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()

        self.DISPLAY = pygame.display.set_mode((1000, 800), 0, 32)
        self.camera = PygameCamera(self.DISPLAY,
                              se2=torch.tensor([500., 500., 0]),
                              scale=torch.ones(2) * 5.5, resolution=(1000, 800))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pygame.quit()
        sys.exit()


def test_car():

    with PygameManager() as pg:

        t = 0.0
        n_cars = 2
        car = Car(n_cars=n_cars)
        car.body.color = RED
        car.body.pos[0, 0] = -20.
        car.body.pos[1, 0] = +20.

        track = Polygon([
            [10., +10, -10, -10],
            [20., -20, -20, +20]
        ], N=1)
        track.color = GRAY

        while True:
            pg.clock.tick(25)
            pg.DISPLAY.fill(WHITE)
            pg.camera.draw(track)
            zero = torch.zeros(n_cars)
            theta1 = torch.full((n_cars, ), fill_value=t % torch.pi)
            theta2 = torch.full((n_cars,), fill_value=t % torch.pi / 2)
            theta3 = torch.full((n_cars,), fill_value=t % torch.pi * 2)
            car.set_path_params(theta1, theta2, theta3)
            car.draw(pg.camera)

            pygame.display.flip()

            t += 0.01
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT:
                    break


def test_forward_kin():

    def joint_transform(theta, length):
        N = theta.shape[0]
        se2 = torch.zeros((N, 3), dtype=torch.float, device=theta.device)
        se2[:, 1] = length
        se2[:, 2] = theta
        return adjoint_matrix(se2)

    def forward_kinematics_dh(theta1, length1, theta2, length2, theta3, length3, theta4, length4):
        T01 = joint_transform(theta1, length1)
        T12 = joint_transform(theta2, length2)
        T23 = joint_transform(theta3, length3)
        T34 = joint_transform(theta4, length4)
        J1 = T01
        J2 = T01.matmul(T12)
        J3 = T01.matmul(T12.matmul(T23))
        J4 = T01.matmul(T12.matmul(T23.matmul(T34)))
        return J1, J2, J3, J4

    def forward_kinematics(theta, length):
        """
        theta (N, J, 1) -> joint angles
        length (N, J, 1) -> arm lengths
        returns a list of length J containing [(N, 3, 3), (N, 3, 3) .... J ] transformation matrices for J joints
        """
        Ts = []
        for t, l in zip(theta.permute(1, 0), length.permute(1, 0)):
            Ts += [joint_transform(t.T, l.T)]
        Js = [Ts[0]]
        for i, T in enumerate(Ts[1:]):
            Js += [Js[-1].matmul(T)]
        return Js

    theta1 = torch.tensor([0.2, 0.2])
    theta2 = torch.tensor([0.2, 0.2])
    theta3 = torch.tensor([0.2, 0.2])
    theta4 = torch.tensor([0.2, 0.2])
    theta = torch.full((2, 4), fill_value=0.2)

    length1 = torch.tensor([0.2, 0.2])
    length2 = torch.tensor([0.2, 0.2])
    length3 = torch.tensor([0.2, 0.2])
    length4 = torch.tensor([0.2, 0.2])
    length = torch.full((2, 4), fill_value=0.2)

    J_dh = forward_kinematics_dh(theta1, length1, theta2, length2, theta3, length3, theta4, length4)
    J = forward_kinematics(theta, length)

    for a, b in zip(J, J_dh):
        assert torch.allclose(a, b)