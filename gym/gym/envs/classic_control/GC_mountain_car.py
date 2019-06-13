"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from gym.utils.gridcell import gridcell

class GC_MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_position = -1.2
        self.max_position = 1.2
        self.max_speed = 0.2
        self.goal_position = 0.5  # to 1 decimal place
        self.goal_width = 0.2  # to 1 decimal place
        self.hillscale = 0
        self.rewarded = False
        self.rewarded_count = 0
        self.velocity_shift = 0.05
        self.obsError = 0
        self.min_visual_input = 0
        self.max_visual_input = 1
        self.trial_count = 0
        self.trialtype = "beaconed"
        self.start_pos = -0.6
        self.visual_input = 0

        self.gridcell = gridcell()

        self.low = np.array([self.min_visual_input, 0])
        self.high = np.array([self.max_visual_input, self.max_speed])

        # add lows and highs for grid cell firing rates
        self.low = np.append(self.low, np.zeros(np.shape(self.gridcell.grid_scales)))
        self.high = np.append(self.high, np.ones(np.shape(self.gridcell.grid_scales)))

        self.viewer = None

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_obs_error(self, obsError):
        self.obsError = obsError

    def SetTrialType(self):
        self.trial_count += 1
        if self.trial_count % 5 == 0:
            self.trialtype = "non_beaconed"
        else:
            self.trialtype = "beaconed"

    def get_visual_input(self, position):

        if position >= self.goal_position - (self.goal_width / 2) and position <= self.goal_position + (
                self.goal_width / 2):
            visual_input = 1
        else:
            visual_input = 0

        return visual_input

    def getTrialType(self):
        return self.trialtype

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity = action*self.velocity_shift      # binary (2 actionspace)
        velocity = np.clip(velocity, 0, self.max_speed)

        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        # observation of visual input is scaled from position 0 from 0 to 1 until end of rewarded zone (0.6)
        if self.trialtype == "beaconed":
            visual_input = self.get_visual_input(position)

        elif self.trialtype == "non_beaconed":
            visual_input = 0

        elif self.trialtype == "probe":
            visual_input = 0

        grid_cell_input = self.gridcell.sine_wave_grid_firing(position)

        done= False
        reward = -1

        if ((position<(self.goal_position+(self.goal_width/2))) and (position>(self.goal_position-(self.goal_width/2)))
                and (np.round(velocity, decimals=2) == 0) and (self.rewarded == False)):
                self.rewarded = True
                reward = 100

        if(position>(self.goal_position+(self.goal_width*2))) and self.rewarded:
            done = True

        self.state = (position, velocity)

        # observation includes binary visual input, velocity and a grid cell readout of location
        obs = np.append(np.array([visual_input, velocity]), grid_cell_input)

        return obs, reward, done, np.array(self.state)

    def reset(self):
        self.state = np.array([self.start_pos, 0])
        #self.obs = np.array([0, 0])
        self.rewarded = False
        self.rewarded_count = 0
        self.SetTrialType()

        obs = np.append(np.array([0, 0]), self.gridcell.sine_wave_grid_firing(self.start_pos))
        return obs

    def _height(self, xs):
            return np.sin(self.hillscale* xs)*.45+.55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            #self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            zonex1 = (self.goal_position-(self.goal_width/2)-self.min_position)*scale
            zonex2 = (self.goal_position+(self.goal_width/2)-self.min_position)*scale
            zoney1 = self._height(self.goal_position)*scale
            zoney2 = zoney1-20
            zone = rendering.FilledPolygon([(zonex1, zoney1),(zonex1, zoney2),(zonex2, zoney2),(zonex2, zoney1)])
            zone.set_color(0,.9,0)
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(zone)
            self.viewer.add_geom(flag)
            self.viewer.add_geom(self.track)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        if self.hillscale == 0:
            self.cartrans.set_rotation(self.hillscale)
        else:
            self.cartrans.set_rotation(math.cos(self.hillscale * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
