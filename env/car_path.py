import pygame, sys
from pygame.locals import *
import math
import numpy as np
import gym
from gym.utils import seeding
import torch
import polygon
from polygon import raycast, line_seg_intersect, edges, clip_line_segment_by_poly, roll, PygameCamera
from math import radians
from copy import deepcopy

params = {
    'CAR_WIDTH': 2.,
    'CAR_LENGTH': 4.,
    'PATH_EXTRA_WIDTH': 1.,
    'PATH_LENGTH': 5.,
    'PATH_SECTIONS': 5,
    'CAR_SCALE': (1.5, 1.5),
    'N_BEAMS': 13,
    'MAX_BEAM_LEN': 300,
    'MAX_STEERING_ANGLE': radians(80.),
    'MAX_STEERING_ADJUST': radians(50.),
    'MAX_TARGET_DISTANCE': 50.,
}


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
            return False, ()  # Failed
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
        return False, ()

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

    return True, (np.array(road_poly), road_colors, np.array(road_left), np.array(road_right), track)


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


ROAD_EDGE = hex_rgb('d6d5c9')
ROAD = hex_rgb('032b43')
BLUE = (0, 0, 255)
GRASS = hex_rgb('136f63')
CAR = hex_rgb('d00000')
PATH_CLEAR = hex_rgb('7ae582')
PATH_BLOCKED = (255, 160, 160)
BEAMS = hex_rgb('ffba08')


class Track:
    def __init__(self, seed, device='cpu'):

        # generate the track
        while True:
            track_created, track_data = create_track(seed=seed)
            if track_created:
                self.track, self.colors, road_left, road_right, self.info = track_data
                break

        self.tarmac = torch.from_numpy(self.track).float().to(device)

        self.max_x = self.tarmac[:, :, 0].max().item()
        self.min_x = self.tarmac[:, :, 0].min().item()
        self.max_y = self.tarmac[:, :, 1].max().item()
        self.min_y = self.tarmac[:, :, 1].min().item()

        road_left, road_right = torch.from_numpy(road_left).float().to(device), torch.from_numpy(road_right).float().to(
            device)

        road_left_end, road_right_end = roll(road_left), roll(road_right)
        road_start = torch.cat((road_left, road_right))
        road_end = torch.cat((road_left_end, road_right_end))

        self.curb_segment_start = road_start
        self.curb_segment_end = road_end

    def get_start_pos(self, i):
        return self.info[i][2], self.info[i][3]

    def get_start_angle(self, i):
        return self.info[i][1]

    @property
    def num_poly(self):
        track_polygons, _, _ = self.tarmac.shape
        return track_polygons

    @property
    def height(self):
        return self.max_y - self.min_y

    @property
    def width(self):
        return self.max_x - self.min_x


class Car:
    def __init__(self, n_cars, device='cpu'):
        self.n_cars = n_cars
        self.device = device
        cw, cl = params['CAR_WIDTH'], params['CAR_LENGTH']
        self.body = polygon.Polygon([
            [cw / 2, cw / 2, -cw / 2, -cw / 2],
            [cl / 2, -cl / 2, -cl / 2, cl / 2]
        ], N=n_cars).to(device)

        W = params['PATH_EXTRA_WIDTH'] + params['CAR_WIDTH']

        self.path = []
        for j in range(params['PATH_SECTIONS'] + 1):
            self.path += [polygon.Polygon([
                [W / 2, W / 2, -W / 2, -W / 2],
                [1., 0, 0, 1.],
            ], N=n_cars).to(device)]
            self.body.attach(self.path[j])

        self.end_effector = polygon.Circle(radius=W * 2, N=n_cars).to(device)
        self.body.attach(self.end_effector)

        lidar_angles = torch.tensor(
            [radians(theta) for theta in torch.linspace(-45., 45, params['N_BEAMS'])]
        )
        self.lidar = polygon.Polygon([
            [0.] * len(lidar_angles) + [math.sin(theta) for theta in lidar_angles],
            [0.] * len(lidar_angles) + [math.cos(theta) for theta in lidar_angles]
        ], N=n_cars).to(device)
        self.body.attach(self.lidar)

        # self.length1 = torch.ones(self.n_cars, device=device)
        self.theta1 = torch.zeros(self.n_cars, device=device)
        self.theta2 = torch.zeros(self.n_cars, device=device)
        self.theta3 = torch.zeros(self.n_cars, device=device)

    @staticmethod
    def joint_transform(theta, length):
        N = theta.shape[0]
        se2 = torch.zeros((N, 3), dtype=torch.float, device=theta.device)
        se2[:, 1] = length
        se2[:, 2] = theta
        return polygon.adjoint_matrix(se2)

    @staticmethod
    def forward_kinematics(theta, length):
        """
        theta (N, J, 1) -> joint angles
        length (N, J, 1) -> arm lengths
        returns a list of length J containing [(N, 3, 3), (N, 3, 3) .... J ] transformation matrices for J joints
        """
        Ts = []
        for t, l in zip(theta.permute(1, 0), length.permute(1, 0)):
            Ts += [Car.joint_transform(t.T, l.T)]
        Js = [Ts[0]]
        for i, T in enumerate(Ts[1:]):
            Js += [Js[-1].matmul(T)]
        return Js

    def set_path_params(self, theta):
        N, J = theta.shape

        # compute the positions of the path components
        zero = torch.zeros((N, 1), device=theta.device)
        theta = torch.cat((theta, zero), dim=1)
        pl = torch.full((N, 1), params['PATH_LENGTH'], device=theta.device)
        length = torch.cat([zero] + [pl] * (params['PATH_SECTIONS']), dim=1)
        one = torch.ones((N, 1), device=theta.device)
        joints = Car.forward_kinematics(theta, length)

        # put the rectangles in the correct locations
        for i in range(params['PATH_SECTIONS']):
            self.path[i].se2 = polygon.se2_from_adjoint(joints[i])
            self.path[i].scale = torch.cat([one, pl], dim=1)
        self.end_effector.se2 = polygon.se2_from_adjoint(joints[params['PATH_SECTIONS']])

    def end_effector_se2(self):
        return self.end_effector.se2

    def path_edges(self):
        # convert the path to a set of line segs in world space
        path_edges = list(zip(*[edges(path.world()) for p, path in enumerate(self.path)]))
        path_edge_start = torch.cat(path_edges[0], dim=1)
        path_edge_end = torch.cat(path_edges[1], dim=1)
        return path_edge_start, path_edge_end

    def check_collision(self, curb_segment_start, curb_segment_end, tarmac):

        # check car path
        path_edge_start, path_edge_end = self.path_edges()

        # check for intersections with the curb
        p, path_intersect_roadside = line_seg_intersect(path_edge_start, path_edge_end,
                                                        curb_segment_start,
                                                        curb_segment_end)
        path_intersects_curb = path_intersect_roadside.permute(1, 0, 2).any(1).any(1)

        start_path_clip, end_path_clip, road_intersect_path = \
            clip_line_segment_by_poly(path_edge_start, path_edge_end, tarmac)

        if road_intersect_path.all(2).any():
            mask = road_intersect_path.all(2)
            print(path_edge_start[mask].shape)
            print(path_edge_start[mask].shape)
            print(path_edge_end[mask])
            print(start_path_clip[mask])
            print(end_path_clip[mask])

        tarmac_segments_traversed = road_intersect_path.any(1)
        return path_intersects_curb, tarmac_segments_traversed

    def draw(self, camera):
        camera.draw(self.body)
        for path in self.path:
            camera.draw(path)


class CarRacingPathEnv(gym.Env):
    def __init__(self, seed=None, n_cars=1, max_episode_steps=None, headless=False, device='cpu', crash_penalty=0.,
                 render_fps=2):
        """
        Vectorized Car pathfinding simulation.

        state_space is a (n_cars * N_BEAMS) tensor of 2D lidar distances originating from the car
        action_space is a (n_cars * 2) tensor of steering angle and distance
        steering angles in range -1..1 and distance in range 0..1
            steering angle is multiplied by d by MAX_STEERING param to get angle
            distance is
        """
        with torch.no_grad():
            self._seed = seed
            self.n_cars = n_cars
            self.max_episode_steps = max_episode_steps
            self.headless = headless
            self.resolution = (1280, 764)
            self.render_fps = render_fps
            self.device = device
            self.crash_penalty = crash_penalty
            self.observation_space = gym.spaces.Box(low=0., high=1., shape=(params['N_BEAMS'],), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1., high=1., shape=(params['PATH_SECTIONS'],), dtype=np.float32)
            if not headless:
                # init pygame DISPLAY and VARS
                pygame.init()
                pygame.font.init()
                pygame.font.get_init()
                self.resolution = (1289, 764)
                self.DISPLAY = pygame.display.set_mode(self.resolution, 0, 32)
                pygame.display.set_caption('Car Racing')
                self.clock = pygame.time.Clock()
            self.reset()

    def seed(self, seed=None):
        self._seed = seed

    def cast_lidar(self):
        # lidar beams
        lidar = self.car.lidar.world()
        beam_origin, beam_end = lidar.chunk(2, dim=1)
        beam_vector = beam_end - beam_origin
        beam_origin, beam_end, beam_len, beam_hit, beam_index = \
            raycast(beam_origin, beam_vector, self.track.curb_segment_start, self.track.curb_segment_end,
                    max_len=params['MAX_BEAM_LEN'])
        return beam_origin, beam_end, beam_len

    def reset(self):
        with torch.no_grad():
            self.n_steps = 0

            self.track = Track(self._seed, device=self.device)
            self.track_visited = torch.zeros((self.n_cars, self.track.num_poly), dtype=torch.bool, device=self.device)
            if not self.headless:
                self.camera = PygameCamera(
                    DISPLAY=self.DISPLAY,
                    se2=torch.tensor([-self.track.min_x * 3.0, -self.track.min_y * 3.0, 0.]),
                    # scale=torch.tensor([self.resolution[0] / self.track.width, self.resolution[1] / self.track.height])
                    scale=torch.ones(2) * 3.0
                ).to(self.device)
            starts = torch.randint(0, len(self.track.info), (self.n_cars,))
            self.car_start_pos = torch.tensor([self.track.get_start_pos(s) for s in starts], device=self.device)
            self.car_start_theta = -torch.tensor([self.track.get_start_angle(s) for s in starts], device=self.device)

            self.car = Car(self.n_cars, device=self.device)
            self.car_prev = Car(self.n_cars, device=self.device) if not self.headless else None

            self.path_off_road = False
            self.road_traversed = None
            self.car.body.pos = self.car_start_pos
            self.car.body.theta = self.car_start_theta
            if not self.headless:
                self.car_prev.body.pos = self.car_start_pos
                self.car_prev.body.theta = self.car_start_theta
            self.beam_origin, self.beam_end, beam_len = self.cast_lidar()
            self.done = torch.zeros(self.n_cars, dtype=torch.bool, device=self.device)
            self.n_steps = 0
            self.respawned = torch.zeros(self.n_cars, dtype=torch.bool, device=self.device)
            self.returns = torch.zeros(self.n_cars, device=self.device)
            self.traj_length = torch.zeros(self.n_cars, device=self.device, dtype=torch.int)
            self.episode_returns = []
            self.episode_length = []
            return beam_len / params['MAX_BEAM_LEN']

    def respawn(self):
        # reset any car that was done
        self.respawned = torch.zeros(self.n_cars, dtype=torch.bool, device=self.device)
        for i, d in enumerate(self.done):
            if d:
                start = torch.randint(0, len(self.track.info), ())
                pos, theta = self.track.get_start_pos(start), -self.track.get_start_angle(start)
                self.car.body.pos[i] = torch.tensor(pos, device=self.device)
                self.car.body.theta[i] = theta
                if not self.headless:
                    self.car_prev.body.pos[i] = torch.tensor(pos, device=self.device)
                    self.car_prev.body.theta[i] = theta
                self.respawned[i] = True

                self.track_visited[self.respawned] = False
                self.road_traversed[self.respawned] = False

                self.episode_returns.append(self.returns[i].item())
                self.episode_length.append(self.traj_length[i].item())
                self.returns[i] = 0.
                self.traj_length[i] = 0
        return self.respawned

    def step(self, action):
        with torch.no_grad():

            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action)

            # steering_input = action[:, 0].clamp(-1, 1)
            # target_dist_input = action[:, 1].clamp(-1, 1)
            # target_dist_input = (target_dist_input + 1.) / 2.  # rescale -1 - 1 to 0 - 1

            # set the steering angles and distance
            # steer_angle = params['MAX_STEERING_ANGLE'] * steering_input
            # curve_apex = params['MAX_TARGET_DISTANCE'] * target_dist_input

            action = action.clamp(-1, 1)
            theta = action * params['MAX_STEERING_ANGLE']
            self.car.set_path_params(theta)

            # self.car.set_path(steer_angle, target_distance)
            # self.car.set_path_params(curve_apex, steer_angle)
            # theta_test = torch.tensor([(self.n_steps / 20.) % (torch.pi * 2)], device=self.device)
            # theta_test = theta_test.repeat(self.n_cars)

            # check if path goes of road, and how many tarmac segments it passes over
            self.path_off_road, self.road_traversed = self.car.check_collision(
                self.track.curb_segment_start, self.track.curb_segment_end, self.track.tarmac)

            if not self.headless:
                # backup the old car so we can display the transition start
                self.car_prev = deepcopy(self.car)

            # move car
            self.car.body.se2 = self.car.end_effector.se2_world

            reward = (self.road_traversed & ~self.track_visited).sum(1) - 0.1
            self.track_visited[self.road_traversed] = True
            lapped = self.track_visited.all(1)
            self.track_visited[lapped] = False  # reset track visited when lapped

            # respawn the guys that died last round
            respawned = self.respawn()
            reward[respawned] = 0.

            # run is permanently done once off road, ie: crashed
            self.done = self.path_off_road | ~self.road_traversed.any(1)
            reward[self.done] = -self.crash_penalty

            # if your respawning, you cant be done
            self.done[respawned] = False

            # we ran out of time, we are all done
            if self.max_episode_steps is not None:
                if self.n_steps > self.max_episode_steps:
                    self.done[:] = True

            # cat the lidar beam from current position
            self.beam_origin, self.beam_end, beam_len = self.cast_lidar()
            state = beam_len / params['MAX_BEAM_LEN']  # normalize the beam_len

            info = {}

            # accumulate stats
            self.n_steps += 1
            self.returns += reward
            self.traj_length += 1
            info['returns'] = self.episode_returns
            info['epi_len'] = self.episode_length
            info['reset'] = self.respawned

            return state, reward, self.done, info

    def draw_track(self):
        self.camera.draw_polygon(self.track.tarmac, self.track.colors[0])
        self.camera.draw_line(self.track.curb_segment_start, self.track.curb_segment_end, color=ROAD_EDGE, width=4)

    def draw_path(self, car):
        if self.road_traversed is not None:
            path_colors = torch.empty((self.path_off_road.shape[0], 3), dtype=torch.uint8)
            path_colors[self.path_off_road] = torch.tensor(PATH_BLOCKED, dtype=torch.uint8)
            path_colors[~self.path_off_road] = torch.tensor(PATH_CLEAR, dtype=torch.uint8)
            for path in car.path:
                self.camera.draw_polygon(path.world(), path_colors)
            self.camera.draw_circle(*car.end_effector.world(), BLUE)

    def render(self, mode='human', fps=None, text=None):
        if not self.headless:
            with torch.no_grad():
                fps = self.render_fps if fps is None else fps

                self.clock.tick(fps)
                self.camera.DISPLAY.fill(GRASS)
                self.draw_track()

                if self.road_traversed is not None:
                    road_traversed = self.road_traversed[~self.respawned]
                    track_traversed = self.track.tarmac[None, ...].repeat(road_traversed.shape[0], 1, 1, 1)[
                        road_traversed]
                    self.camera.draw_polygon(track_traversed, BLUE)

                # draw car state before transition
                self.camera.draw_polygon(self.car_prev.body.world(), CAR)
                # self.draw_car_path(self.car_prev)
                self.draw_path(self.car_prev)

                returns_str = 'retns: '
                for r in self.returns[-5:]:
                    returns_str += f'{r:3.0f},'

                epi_len_str = 'e_len:'
                for r in self.traj_length[-5:]:
                    epi_len_str += f'{r:3},'

                display_str = [returns_str, epi_len_str]

                self.camera.draw_text(display_str, color=(0, 255, 0))
                if text is not None:
                    self.camera.draw_text(text, topleft=(800, 700), color=(0, 255, 0))

                pygame.display.flip()

                self.clock.tick(fps)
                self.camera.DISPLAY.fill(GRASS)
                self.draw_track()

                # draw car state after transition
                self.camera.draw_polygon(self.car.body.world(), CAR)

                # draw beams
                self.camera.draw_line(self.beam_origin[~self.done], self.beam_end[~self.done], BEAMS, width=1)
                # self.draw_path(self.car)

                self.camera.draw_text(display_str, color=(0, 255, 0))
                if text is not None:
                    self.camera.draw_text(text, topleft=(800, 700), color=(0, 255, 0))

                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()


def main():
    N_CARS = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = CarRacingPathEnv(seed=3, n_cars=N_CARS, device=device)

    reward_total = torch.zeros(N_CARS, device=device)

    def policy(state):
        with torch.no_grad():
            # simple steering angle policy with conservative moves
            dist, index = torch.max(state, dim=1)
            n_max = (state == dist.unsqueeze(1)).sum(dim=1)
            index += n_max.div(2, rounding_mode='trunc')
            turn = index - params['N_BEAMS'] // 2
            turn = turn * 1. / 8.
            dist = dist * params['MAX_BEAM_LEN'] / params['MAX_TARGET_DISTANCE'] - 1.
            return torch.stack([turn / params['PATH_SECTIONS']] * params['PATH_SECTIONS'], dim=-1)

    state = env.reset()
    env.render()
    done = torch.zeros(N_CARS, dtype=torch.bool)
    # while not done.all():
    while True:
        action = policy(state)
        state, reward, done, info = env.step(action)
        reward_total += reward
        env.render(fps=0.5, text=['hello'])


if __name__ == '__main__':
    main()
