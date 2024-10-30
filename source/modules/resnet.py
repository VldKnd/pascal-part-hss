import torch


class ResNetBlock(torch.nn.Module):

    def __init__(self, in_channels: int, inner_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=inner_channels,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn1 = torch.nn.BatchNorm2d(num_features=inner_channels)
        self.relu = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(
            in_channels=inner_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)

        self.projection = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.projection(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out