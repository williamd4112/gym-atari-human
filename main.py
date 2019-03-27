import gym
import argparse
import time
import pickle

# Copy from OpenAI baseline: https://github.com/openai/baselines
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='Environment', type=str, default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--delay', help='Delay time at each frame', type=float, default=0.03)
    args = parser.parse_args()

    env = gym.make(args.env)
    env = NoopResetEnv(env)
    env = EpisodicLifeEnv(env)
    env.render()

    exit = False    
    action_meaning = {key: idx for idx, key in enumerate(env.unwrapped.get_action_meanings())}
    human_action = action_meaning['NOOP']    

    key_mapping = { 0xff51: ('LEFT', 1),
                    0xff52: ('UP', 0),
                    0xff53: ('RIGHT', 1),
                    0xff54: ('DOWN', 0),
                    ord('z'): ('FIRE', 2)}

    # Ordered by the action meaning
    key_buffer = [  '', # UP/DOWN
                    '', # LEFT/RIGHT
                    '' # FIRE
                ]

    def get_act_from_key(key):
        try:
            a = key_mapping[key]
        except KeyError:
            print('Unknown key ', key)
            return None
        return a

    def convert_key_buffer_to_atari_action():
        action = ''.join(key_buffer)
        if action == '':
            return action_meaning['NOOP']
        else:
            try:
                return action_meaning[action]
            except KeyError:
                print('Invalid key')
                return action_meaning['NOOP']

    def key_press(key, mod):
        global exit
        if int(key) == 32:
            exit = True
            return  
        a = get_act_from_key(key)
        if a == None: return
        key_buffer[a[1]] = a[0]
        
    def key_release(key, mod):
        a = get_act_from_key(key)
        if a == None: return
        key_buffer[a[1]] = ''

    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    
    print('Arrow keys: UP/DOWN/LEFT/RIGHT; Z: FIRE')
    print('Press space to exit.')    
    replay = []
    s = env.reset()
    episode_count = 0
    while True:
        env.render()
        a = convert_key_buffer_to_atari_action()       
        stp_1, r, d ,_ = env.step(a)

        replay.append((s, r, d, a, stp_1))
        s = stp_1

        time.sleep(args.delay)
        if d:
            print('Episode %d done; Timesteps: %d' % (episode_count, len(replay)))
            env.reset()
            episode_count += 1

        if exit:
            break
    
    output_path = '%s-demo.pkl' % (args.env)
    with open(output_path, 'wb') as f:
        pickle.dump(replay, f)
    print('Demonstration data is saved at %s.' % (output_path))


