import numpy as np

from DQN.replay_buffer import ReplayBuffer


def test_replay_buffer_push_sample_len():
    buffer = ReplayBuffer(buffer_size=10)
    for i in range(5):
        s = np.array([float(i)] * 8, dtype=np.float32)
        buffer.push(s, i, float(i), s, False)

    assert len(buffer) == 5
    batch = buffer.sample(3)
    states, _actions, _r, _ns, _d = batch
    assert states.shape[0] == 3
