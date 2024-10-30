import torch


class View(torch.nn.Module):

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, input: torch.Tensor):
        return input.view(-1, *self.shape)