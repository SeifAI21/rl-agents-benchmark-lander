import torch

from DQN.networks import DeepQNetwork


def test_deep_qnetwork_output_shape():
    model = DeepQNetwork(8, 4, hidden_size=64)
    dummy_state = torch.randn(1, 8)
    output = model(dummy_state)
    assert output.shape == (1, 4)
