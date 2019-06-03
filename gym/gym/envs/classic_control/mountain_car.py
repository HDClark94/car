"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class MountainCarEnv(gym.Env):
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
        self.trialtype = "non_beaconed"


        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        # change action space between 2 and 3 for binary vs continous velocity
        self.actiondim = 2
        self.action_space = spaces.Discrete(self.actiondim)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_obs_error(self, obsError):
        self.obsError = obsError

    def set_action_dim(self, actionDim):
        self.actiondim = actionDim
        self.action_space = spaces.Discrete(self.actiondim)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity = action*self.velocity_shift      # binary (2 actionspace)
        velocity = np.clip(velocity, 0, self.max_speed)

        self.obs[0] += velocity+(action*self.np_random.normal(0, self.obsError))
        self.obs[0] = np.clip(self.obs[0], self.min_position, self.max_position)

        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        done= False
        reward = -1

                # reward option one ()
        if ((position<(self.goal_position+(self.goal_width/2))) and (position>(self.goal_position-(self.goal_width/2))) and (np.round(velocity, decimals=2) == 0) and (self.rewarded == False)):
                #print("rewarded and the postion =", position)
                self.rewarded = True
                #done = True
                reward = 100

        if(position>(self.goal_position+(self.goal_width*2))) and self.rewarded:
            done = True

        self.state = (position, velocity)

        self.obs[1] = self.state[1] # velocity is same
        #self.obs[0] = 0

        return np.array(self.obs), reward, done, np.array(self.state)

    def reset(self):
        self.state = np.array([-0.6, 0])
        self.obs = self.state
        self.rewarded = False
        self.rewarded_count = 0
        return np.array(self.state)

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
