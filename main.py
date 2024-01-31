import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import load_data, create_datasets
from model import MultimodalModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for text, image, labels in train_loader:
        text, image, labels = text.to(device), image.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(text, image)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for text, image, labels in val_loader:
            text, image, labels = text.to(device), image.to(device), labels.to(device)
            outputs = model(text, image)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def main(args):
    train_df, val_df, test_df, labels = load_data(args.train_path, args.test_path)
    train_dataset, val_dataset, test_dataset = create_datasets(train_df, val_df, test_df, labels)

    # 创建数据加载器
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 创建模型实例
    model = MultimodalModel(use_text=args.use_text, use_image=args.use_image).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_accuracy = validate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {train_loss}, Validation Accuracy: {val_accuracy}")

    # 消融实验
    if not args.no_text_experiment:
        model_no_text = MultimodalModel(use_text=False, use_image=args.use_image).to(device)
        optimizer_no_text = optim.Adam(model_no_text.parameters(), lr=args.learning_rate)
        for epoch in range(args.num_epochs):
            train_loss = train(model_no_text, train_loader, criterion, optimizer_no_text, device)
            val_accuracy = validate(model_no_text, val_loader, criterion, device)
            print(
                f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {train_loss}, Validation Accuracy (No Text): {val_accuracy}")

    if not args.no_image_experiment:
        model_no_image = MultimodalModel(use_text=args.use_text, use_image=False).to(device)
        optimizer_no_image = optim.Adam(model_no_image.parameters(), lr=args.learning_rate)
        for epoch in range(args.num_epochs):
            train_loss = train(model_no_image, train_loader, criterion, optimizer_no_image, device)
            val_accuracy = validate(model_no_image, val_loader, criterion, device)
            print(
                f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {train_loss}, Validation Accuracy (No Image): {val_accuracy}")


    # 预测和生成结果
    model.eval()
    test_predictions = []

    with torch.no_grad():
        for text, image, in test_loader:
            text, image = text.to(device), image.to(device)

            outputs = model(text, image)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.cpu().numpy())

    # 将预测结果转换为标签
    label_encoder = LabelEncoder()
    predicted_labels = label_encoder.inverse_transform(test_predictions)

    # 生成结果文件
    result_df = pd.DataFrame({"guid": test_df["guid"], "label": predicted_labels})
    result_df.to_csv(args.result_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Model Training and Evaluation")
    parser.add_argument("--train_path", type=str, default="data/train.txt", help="Path to the training data file")
    parser.add_argument("--test_path", type=str, default="data/test_without_label.txt", help="Path to the test data file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--use_text", action="store_true", help="Include text data in the model")
    parser.add_argument("--use_image", action="store_true", help="Include image data in the model")
    parser.add_argument("--no_text_experiment", action="store_true", help="Run experiment without text data")
    parser.add_argument("--no_image_experiment", action="store_true", help="Run experiment without image data")
    parser.add_argument("--result_path", type=str, default="test_results.csv",
                        help="Path to save test results CSV file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    main(args)
