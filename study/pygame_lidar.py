import pygame, sys
from pygame.locals import *
import math
import numpy as np
import gym
from gym.utils import seeding
import torch
import polygon
from polygon import raycast, line_seg_intersect, edges, clip_line_segment_by_poly, roll, transform_matrix, \
    apply_transform
from math import radians
from copy import copy


def create_track(seed):
    np_random, seed = seeding.np_random(seed)
    CHECKPOINTS = 12
    SCALE = 6.0  # Track scale
    TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
    TRACK_DETAIL_STEP = 21 / SCALE
    TRACK_TURN_RATE = 0.31
    TRACK_WIDTH = 40 / SCALE
    BORDER = 8 / SCALE
    BORDER_MIN_COUNT = 4

    ROAD_COLOR = [0.4, 0.4, 0.4]
    start_alpha = None
    road = []
    road_colors = []
    road_poly = []
    verbose = 0
    fd_tile = None

    # Create checkpoints
    checkpoints = []
    for c in range(CHECKPOINTS):
        noise = np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
        alpha = 2 * math.pi * c / CHECKPOINTS + noise
        rad = np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

        if c == 0:
            alpha = 0
            rad = 1.5 * TRACK_RAD
        if c == CHECKPOINTS - 1:
            alpha = 2 * math.pi * c / CHECKPOINTS
            start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
            rad = 1.5 * TRACK_RAD

        checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

    # Go from one checkpoint to another to create track
    x, y, beta = 1.5 * TRACK_RAD, 0, 0
    dest_i = 0
    laps = 0
    track = []
    no_freeze = 2500
    visited_other_side = False
    while True:
        alpha = math.atan2(y, x)
        if visited_other_side and alpha > 0:
            laps += 1
            visited_other_side = False
        if alpha < 0:
            visited_other_side = True
            alpha += 2 * math.pi

        while True:  # Find destination from checkpoints
            failed = True

            while True:
                dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                if alpha <= dest_alpha:
                    failed = False
                    break
                dest_i += 1
                if dest_i % len(checkpoints) == 0:
                    break

            if not failed:
                break

            alpha -= 2 * math.pi
            continue

        r1x = math.cos(beta)
        r1y = math.sin(beta)
        p1x = -r1y
        p1y = r1x
        dest_dx = dest_x - x  # vector towards destination
        dest_dy = dest_y - y
        # destination vector projected on rad:
        proj = r1x * dest_dx + r1y * dest_dy
        while beta - alpha > 1.5 * math.pi:
            beta -= 2 * math.pi
        while beta - alpha < -1.5 * math.pi:
            beta += 2 * math.pi
        prev_beta = beta
        proj *= SCALE
        if proj > 0.3:
            beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
        if proj < -0.3:
            beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
        x += p1x * TRACK_DETAIL_STEP
        y += p1y * TRACK_DETAIL_STEP
        track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
        if laps > 4:
            break
        no_freeze -= 1
        if no_freeze == 0:
            break

    # Find closed loop range i1..i2, first loop should be ignored, second is OK
    i1, i2 = -1, -1
    i = len(track)
    while True:
        i -= 1
        if i == 0:
            return False  # Failed
        pass_through_start = (
                track[i][0] > start_alpha and track[i - 1][0] <= start_alpha
        )
        if pass_through_start and i2 == -1:
            i2 = i
        elif pass_through_start and i1 == -1:
            i1 = i
            break
    if verbose == 1:
        print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
    assert i1 != -1
    assert i2 != -1

    track = track[i1: i2 - 1]

    first_beta = track[0][1]
    first_perp_x = math.cos(first_beta)
    first_perp_y = math.sin(first_beta)
    # Length of perpendicular jump to put together head and tail
    well_glued_together = np.sqrt(
        np.square(first_perp_x * (track[0][2] - track[-1][2]))
        + np.square(first_perp_y * (track[0][3] - track[-1][3]))
    )
    if well_glued_together > TRACK_DETAIL_STEP:
        return False

    # Red-white border on hard turns
    border = [False] * len(track)
    for i in range(len(track)):
        good = True
        oneside = 0
        for neg in range(BORDER_MIN_COUNT):
            beta1 = track[i - neg - 0][1]
            beta2 = track[i - neg - 1][1]
            good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
            oneside += np.sign(beta1 - beta2)
        good &= abs(oneside) == BORDER_MIN_COUNT
        border[i] = good
    for i in range(len(track)):
        for neg in range(BORDER_MIN_COUNT):
            border[i - neg] |= border[i]

    # Create tiles
    for i in range(len(track)):
        alpha1, beta1, x1, y1 = track[i]
        alpha2, beta2, x2, y2 = track[i - 1]
        road1_l = (
            x1 - TRACK_WIDTH * math.cos(beta1),
            y1 - TRACK_WIDTH * math.sin(beta1),
        )
        road1_r = (
            x1 + TRACK_WIDTH * math.cos(beta1),
            y1 + TRACK_WIDTH * math.sin(beta1),
        )
        road2_l = (
            x2 - TRACK_WIDTH * math.cos(beta2),
            y2 - TRACK_WIDTH * math.sin(beta2),
        )
        road2_r = (
            x2 + TRACK_WIDTH * math.cos(beta2),
            y2 + TRACK_WIDTH * math.sin(beta2),
        )
        # vertices = [road1_l, road1_r, road2_r, road2_l]
        # fd_tile.shape.vertices = vertices
        # t = self.world.CreateStaticBody(fixtures=self.fd_tile)
        # t.userData = t
        c = 0.01 * (i % 3)
        # t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
        color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
        color = [(c * 255) // 1 for c in color]
        # t.road_visited = False
        # t.road_friction = 1.0
        # t.fixtures[0].sensor = True
        road_poly.append([road1_l, road1_r, road2_r, road2_l])
        road_colors.append(color)
        # self.road.append(t)
        # if border[i]:
        #     side = np.sign(beta2 - beta1)
        #     b1_l = (
        #         x1 + side * TRACK_WIDTH * math.cos(beta1),
        #         y1 + side * TRACK_WIDTH * math.sin(beta1),
        #     )
        #     b1_r = (
        #         x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
        #         y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
        #     )
        #     b2_l = (
        #         x2 + side * TRACK_WIDTH * math.cos(beta2),
        #         y2 + side * TRACK_WIDTH * math.sin(beta2),
        #     )
        #     b2_r = (
        #         x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
        #         y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
        #     )
        #     road_poly.append(
        #         ([b1_l, b1_r, b2_r, b2_l], (255, 255, 255) if i % 2 == 0 else (255, 0, 0))
        #     )
    # self.track = track

    road_left = []
    road_right = []

    # Create 2 non-convex polygons for the inner and outer track
    for i in range(len(track)):
        alpha1, beta1, x1, y1 = track[i]
        road_left += [(
            x1 - TRACK_WIDTH * math.cos(beta1),
            y1 - TRACK_WIDTH * math.sin(beta1),
        )]
        road_right += [(
            x1 + TRACK_WIDTH * math.cos(beta1),
            y1 + TRACK_WIDTH * math.sin(beta1),
        )]

    return np.array(road_poly), road_colors, np.array(road_left), np.array(road_right), track


WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
LIGHT_GREEN = (160, 255, 180)
RED = (255, 0, 0)
LIGHT_RED = (255, 160, 160)


class Camera:
    def __init__(self, se2, scale):
        self.se2 = se2 if len(se2.shape) == 2 else se2.unsqueeze(0)
        self.scale = scale if len(scale.shape) == 2 else scale.unsqueeze(0)

    def transform(self, verts):
        t_matrix = transform_matrix(self.se2, self.scale)
        return apply_transform(t_matrix, verts)


class PygameCamera(Camera):
    def __init__(self, se2, scale, resolution=(1280, 764)):
        super().__init__(se2, scale)
        self.DISPLAY = pygame.display.set_mode(resolution, 0, 32)

    def draw_polygon(self, model, color):
        car_verts = self.transform(model)
        for i in range(len(model)):
            pygame.draw.polygon(self.DISPLAY, color, car_verts[i].tolist())

    def draw_line(self, origin, end, color, width=1):
        origin, end = origin.flatten(0, -2), end.flatten(0, -2)
        origin, end = self.transform(origin).squeeze(0), self.transform(end).squeeze(0)
        for o, e in zip(origin, end):
            pygame.draw.line(self.DISPLAY, color, o.tolist(), e.tolist(), width=width)


params = {
    'CAR_WIDTH': 4.,
    'CAR_LENGTH': 6.,
    'CAR_SCALE': (1.5, 1.5),
    'N_BEAMS': 13,
    'MAX_BEAM_LEN': 40,
    'MAX_STEERING_ANGLE': radians(50.),
    'MAX_TARGET_DISTANCE': 100.,
}


class CarRacingPathEnv(gym.Env):
    def __init__(self, seed=0, n_cars=1):
        self.n_cars = n_cars

        # init pygame DISPLAY and VARS
        pygame.init()
        self.clock = pygame.time.Clock()
        self.camera = PygameCamera(
            se2=torch.tensor([450., 400., 0.]),
            scale=torch.tensor([3.2, 3.2])
        )

        self.track, self.track_colors, road_left, road_right, track_info = create_track(seed=seed)

        def get_start_pos(i, track_info):
            return track_info[i][2], track_info[i][3]

        def get_start_angle(i, track_info):
            return track_info[i][1]

        self.world_track = torch.from_numpy(self.track).float()

        road_left, road_right = torch.from_numpy(road_left).float(), torch.from_numpy(road_right).float()

        road_left_end, road_right_end = roll(road_left), roll(road_right)
        road_start = torch.cat((road_left, road_right))
        road_end = torch.cat((road_left_end, road_right_end))

        self.road_start = road_start
        self.road_end = road_end

        starts = torch.randint(0, len(track_info), (n_cars,))
        self.car_start_pos = torch.tensor([get_start_pos(s, track_info) for s in starts])
        self.car_start_theta = -torch.tensor([get_start_angle(s, track_info) for s in starts])

        cw, cl = params['CAR_WIDTH'], params['CAR_LENGTH']
        self.car = polygon.Model([
            [cw / 2, cw / 2, -cw / 2, -cw / 2],
            [cl / 2, -cl / 2, -cl / 2, cl / 2]
        ], N=n_cars)

        lidar_angles = torch.tensor(
            [radians(theta) for theta in torch.linspace(-45., 45, params['N_BEAMS'])]
        )

        self.car_lidar = polygon.Model([
            [0.] * len(lidar_angles) + [math.sin(theta) for theta in lidar_angles],
            [0.] * len(lidar_angles) + [math.cos(theta) for theta in lidar_angles]
        ], N=n_cars)

        # self.car_lidar = polygon.Model([
        #     [0.] + [0],
        #     [0.] + [1.]
        # ], N=n_cars)

        # self.car_lidar.pos = torch.tensor([[0., params['CAR_LENGTH']/2] for _ in range(n_cars)])
        self.car.attach(self.car_lidar)

        self.car_path = polygon.Model([
            [cw / 2., cw / 2, -cw / 2, -cw / 2],
            [1., 0., 0., 1.]
        ], N=n_cars)
        self.car.attach(self.car_path)

        self.car_prev = self.car.se2.detach().clone()

        self.path_off_road = False
        self.beam_origin, self.beam_end = None, None
        self.road_intersect_path = None

    def cast_lidar(self):
        # lidar beams
        N, B = self.n_cars, params['N_BEAMS']
        lidar = self.car_lidar.world_verts()
        beam_origin, beam_end = lidar.chunk(2, dim=1)
        beam_vector = beam_end - beam_origin
        beam_origin = beam_origin.flatten(end_dim=-2)
        beam_vector = beam_vector.flatten(end_dim=-2)
        ray_origin, ray_end, ray_len = raycast(beam_origin, beam_vector, self.road_start, self.road_end,
                                               max_len=params['MAX_BEAM_LEN'])
        return ray_origin.unflatten(0, (N, B)), ray_end.unflatten(0, (N, B)), ray_len.unflatten(0, (N, B))

    def reset(self):
        self.path_off_road = False
        self.road_intersect_path = None
        self.car_path.se2[:, :] = 0.
        self.car_path.scale[:, :] = 1.

        self.car.pos = self.car_start_pos
        self.car.theta = self.car_start_theta
        self.car_prev = copy(self.car)
        self.beam_origin, self.beam_end, beam_len = self.cast_lidar()
        return beam_len / params['MAX_BEAM_LEN']

    def step(self, action):

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        # set the steering angles and distance
        steer_angle = params['MAX_STEERING_ANGLE'] * action[:, 0]
        target_distance = params['MAX_TARGET_DISTANCE'] * action[:, 1]
        self.car_path.se2[:, 2] = steer_angle
        self.car_path.scale[:, 1] = target_distance
        self.car_path.pos[:, 1] = params['CAR_LENGTH'] /2.

        # check car_path
        path_edge_start, path_edge_end = edges(self.car_path.world_verts())
        path_edge_start, path_edge_end = path_edge_start.flatten(end_dim=-2), path_edge_end.flatten(end_dim=-2)
        p, path_intersect_roadside = line_seg_intersect(path_edge_start, path_edge_end, self.road_start, self.road_end)
        path_intersect_roadside = path_intersect_roadside.unflatten(1, (self.n_cars, 4))
        # if any of the boundaries of the box intersect the edge, the cars path is not clear
        self.path_off_road = path_intersect_roadside.permute(1, 0, 2).any(1).any(1)

        # detect road polygons in path
        # select_l_r_edges = torch.tensor([0, 2])
        # path_lr_start, path_lr_end = path_edge_start[select_l_r_edges], path_edge_end[select_l_r_edges]
        # start_path_clip, end_path_clip, self.road_intersect_path = clip_line_segment_by_poly(path_lr_start, path_lr_end,
        #                                                                                     self.world_track)

        # move car
        # self.car_prev = copy(self.car)
        # self.car.pos += torch.zeros_like(self.)
        # self.car.theta += steer_angle

        # return agent inputs
        self.beam_origin, self.beam_end, beam_len = self.cast_lidar()
        state = beam_len / params['MAX_BEAM_LEN'] # normalize the beam_len
        # reward = self.road_intersect_path.sum() - 0.1
        done = self.path_off_road
        info = {}
        return state, 0., done, info

    def draw_track(self):
        self.camera.draw_polygon(self.world_track, self.track_colors[0])
        self.camera.draw_line(self.road_start, self.road_end, color=WHITE, width=4)

    def render(self, mode='human', fps=2):

        self.clock.tick(fps)
        self.camera.DISPLAY.fill(LIGHT_GREEN)
        self.draw_track()

        # if self.road_intersect_path is not None:
        #     # draw road highlights
        #     for road_intersect_path, poly in zip(self.road_intersect_path.permute(1, 0), self.world_track):
        #         if road_intersect_path.any():
        #             pygame.draw.polygon(self.DISPLAY, BLUE, poly.tolist())

        # draw car
        # car_verts = self.car.world_verts()
        self.camera.draw_polygon(self.car.world_verts(), RED)
        self.camera.draw_polygon(self.car_path.world_verts(), BLUE)

        # draw car path
        # path_color = LIGHT_RED if self.path_off_road else GREEN
        # pygame.draw.polygon(self.DISPLAY, path_color, self.path.pygame_world_verts)
        # for s, e in zip(path_lr_start, path_lr_end):
        #     pygame.draw.line(DISPLAY, WHITE, s.tolist(), e.tolist())

        pygame.display.flip()

        self.clock.tick(fps)
        self.camera.DISPLAY.fill(LIGHT_GREEN)
        self.draw_track()
        self.camera.draw_polygon(self.car.world_verts(), RED)

        # draw car
        # pygame.draw.polygon(self.DISPLAY, RED, self.car.pygame_world_verts)

        # draw beams
        self.camera.draw_line(self.beam_origin, self.beam_end, YELLOW, width=1)
        # if self.beam_origin is not None:
        #     for o, e in zip(self.beam_origin, self.beam_end):
        #         pygame.draw.line(self.DISPLAY, YELLOW, o.tolist(), e.tolist(), width=2)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()


def main():
    env = CarRacingPathEnv(seed=3, n_cars=2)

    def policy(state):
        # simple steering angle policy with conservative moves
        dist, index = torch.max(state, dim=1)
        n_max = (state == dist.unsqueeze(1)).sum(dim=1)
        index += n_max.div(2, rounding_mode='trunc')
        turn = index - params['N_BEAMS'] // 2
        turn = turn * 1. / 8.
        dist = dist/2. * params['MAX_BEAM_LEN'] / params['MAX_TARGET_DISTANCE']
        return torch.stack([turn, dist], dim=-1)

    state = env.reset()
    env.render()
    while True:
        action = policy(state)
        state, reward, done, info = env.step(action)
        env.render()


if __name__ == '__main__':
    from time import sleep

    with torch.no_grad():
        main()
