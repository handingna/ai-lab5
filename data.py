import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
import torch


class MultimodalDataset(Dataset):
    def __init__(self, text_data, image_folder, labels, tokenizer, transform=None):
        self.text_data = text_data
        self.image_folder = image_folder
        self.labels = labels
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        guid = self.text_data.iloc[idx, 0]

        img_filename = str(guid) + ".jpg"
        img_path = os.path.join(self.image_folder, img_filename)
        image = Image.open(img_path).convert("RGB")

        # 获取文本序列
        text_filename = str(guid) + ".txt"
        text_file_path = os.path.join(self.image_folder, text_filename)

        with open(text_file_path, 'r', encoding='utf-8', errors='replace') as text_file:
            text_sequence = text_file.read().strip()

        # 使用 tokenizer 将文本序列转换为整数序列
        tokenized_text = self.tokenizer(text_sequence, padding='max_length', truncation=True, max_length=128,
                                        return_tensors='pt')
        input_ids = tokenized_text['input_ids'].squeeze()  # 去除额外的维度

        # 添加调整大小的 transform
        if self.transform:
            image = self.transform(image)

        label = torch.as_tensor(self.labels[idx], dtype=torch.long)

        return input_ids, image, label


class TestMultimodalDataset(Dataset):
    def __init__(self, text_data, image_folder, tokenizer, transform=None):
        self.text_data = text_data
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        guid = self.text_data.iloc[idx, 0]

        img_filename = str(guid) + ".jpg"
        img_path = os.path.join(self.image_folder, img_filename)
        image = Image.open(img_path).convert("RGB")

        # 获取文本序列
        text_filename = str(guid) + ".txt"
        text_file_path = os.path.join(self.image_folder, text_filename)

        with open(text_file_path, 'r', encoding='utf-8', errors='replace') as text_file:
            text_sequence = text_file.read().strip()

        # 使用 tokenizer 将文本序列转换为整数序列
        tokenized_text = self.tokenizer(text_sequence, padding='max_length', truncation=True, max_length=128,
                                        return_tensors='pt')
        input_ids = tokenized_text['input_ids'].squeeze()  # 去除额外的维度

        # 添加调整大小的 transform
        if self.transform:
            image = self.transform(image)

        return input_ids, image,


# 使用 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# 数据处理
def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file, sep=",", skiprows=1, names=["guid", "label"])
    test_df = pd.read_csv(test_file, sep=",", skiprows=1, names=["guid", "label"])

    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["label"])
    labels = torch.as_tensor(train_df["label"].values)

    # 划分训练集和验证集
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    return train_df, val_df, test_df, labels


def create_datasets(train_df, val_df, test_df, labels):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = MultimodalDataset(text_data=train_df, image_folder="data/data", labels=labels, tokenizer=tokenizer,
                                      transform=transform)
    val_dataset = MultimodalDataset(text_data=val_df, image_folder="data/data", labels=labels, tokenizer=tokenizer,
                                    transform=transform)
    test_dataset = TestMultimodalDataset(text_data=test_df, image_folder="data/data", tokenizer=tokenizer,
                                         transform=transform)

    return train_dataset, val_dataset, test_dataset
