import torch
from model import EmotionCNN

model = EmotionCNN()

x = torch.randn(1, 3, 96, 96)

y = model(x)

print(y.shape)
