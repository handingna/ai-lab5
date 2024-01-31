import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch


class MultimodalModel(nn.Module):
    def __init__(self, text_embedding_dim=100, num_classes=3, use_text=True, use_image=True):
        super(MultimodalModel, self).__init__()

        self.use_text = use_text
        self.use_image = use_image

        # 文本模型
        if self.use_text:
            self.embedding = nn.Embedding(num_embeddings=50000, embedding_dim=text_embedding_dim)
            self.lstm = nn.LSTM(input_size=text_embedding_dim, hidden_size=50, batch_first=True)

        # 图像模型
        if self.use_image:
            self.image_model = models.resnet18(pretrained=True)
            in_features = self.image_model.fc.in_features
            self.image_model.fc = nn.Identity()

        # 融合层
        if self.use_text and self.use_image:
            self.fc1 = nn.Linear(50 + in_features, 64)
        elif self.use_text:
            self.fc1 = nn.Linear(50, 64)
        elif self.use_image:
            self.fc1 = nn.Linear(in_features, 64)

        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, text, image):
        if self.use_text:
            text_embedding = self.embedding(text)
            _, (text_output, _) = self.lstm(text_embedding)
            text_hidden_state = text_output[-1]

        if self.use_image:
            image_output = self.image_model(image)

        if self.use_text and self.use_image:
            merged_features = torch.cat((text_hidden_state, image_output), dim=1)
        elif self.use_text:
            merged_features = text_hidden_state
        elif self.use_image:
            merged_features = image_output

        x = F.relu(self.fc1(merged_features))
        x = self.fc2(x)

        return x
