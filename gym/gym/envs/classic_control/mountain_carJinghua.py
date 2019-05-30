import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class MountainCarJinghuaEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_position = 0
        self.max_position = 1
        self.max_speed = 0.3
        self.goal_position = 0.5
        self.trialtype = "beaconed"
        self.trial_count = 0

        self.low = np.array([self.min_position, 0])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(self.low, self.high)

        self.seed()
        self.reset()

    def set_obs_error(self, obsError):
        obsError1 = obsError
        # ignore this, only used for pass (doesn't actually do anything functional)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def SetTrialType(self):
        self.trial_count += 1
        if self.trial_count % 5 == 0 and self.trial_count>500:
            self.trialtype = "non_beaconed"
        else:
            self.trialtype = "beaconed"

    def step(self,action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        indicator, velocity = self.state

        position = self.actual[0]

        velocity = (action)*0.05
        #velocity = np.clip(velocity, 0, self.max_speed)
        position += velocity
        position = np.clip(position, 0, self.max_position)

        if position >= self.goal_position -0.05 and position <= self.goal_position+0.05 and self.trialtype == "beaconed":
           indicator = 1
        else:
           indicator = 0
        
        if position >= self.goal_position -0.05 and position <= self.goal_position+0.05 and (np.round(velocity, decimals=2) == 0):
           done = 1
           reward = 10
        elif position == 1:
           done = 2
           reward = 0
        else:
           done = 0
           reward = -0.005

        self.state = (indicator, velocity)

        self.actual = (position, velocity)
        return np.array(self.state), reward, done, np.array(self.actual)

    def reset(self):
        self.state = np.array([0, 0])#np.array([self.np_random.uniform(low=0, high=0.05), 0])
        self.actual = np.array([0, 0])
        self.SetTrialType()
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

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
            ys = np.linspace(0.2,0.2, 100)#self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

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
            flagy1 = 75#self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        #self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def _height(self, xs):
        return 0.2#np.sin(3 * xs)*.45+.55


