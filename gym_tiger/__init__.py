from gym.envs.registration import register

register(
    id='Tiger-v0',
    entry_point='gym_tiger.envs:TigerEnv')
