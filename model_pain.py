import torch
import torch.nn as nn


class PainCNN(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Output becomes 256 × 4 × 4 for 64×64 input
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256 * 4 * 4,
                512
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = PainCNN()

    sample = torch.randn(
        1,
        3,
        64,
        64
    )

    output = model(sample)

    print(model)

    print("\nOutput Shape:", output.shape)