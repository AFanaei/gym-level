import numpy as np


def step(env, change_action_index, ep_len):
    e = env()

    # for x in range(e.action_space.shape[0]):
    inp = np.zeros((e.action_space.shape[0], ep_len))
    inp[change_action_index, 1] = 5
    e.reset()
    e.render()
    for i in range(ep_len):
        e.step(inp[:, i])
        e.render()
