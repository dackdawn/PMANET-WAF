import os
import torch
import torch.nn.functional as F
from torch.utils.data import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from data_processing import  dataPreprocessFromCSV
from Model_PMA import Model, CharBertModel
import numpy as np
import time

IS_CHARBERT = False

# 定义类别信息字典
CLASSES_DICT = {
    0: {"name": "benign"},
    1: {"name": "malware"},
    # 2: {"name": "phishing"},
    # 3: {"name": "defacement"},
    # 可以根据需要添加更多类别
}

# 类别数量
NUM_CLASSES = len(CLASSES_DICT)

def test_binary(model, device, test_loader, output_name):
    """
    Perform binary classification testing using the given model.

    :param model: The model for binary classification.
    :param device: The device to run testing on (e.g., CPU or GPU).
    :param test_loader: The data loader for test data.
    :return: A tuple containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []  # Save predicted probabilities
    
    start_time = time.time()
    if IS_CHARBERT:
        i = 0
        for batch_idx, (x1, x2, x3, x4, x5, x6, y) in enumerate(test_loader):
            i += 1
            x1, x2, x3, x4, x5, x6, y = x1.to(device), x2.to(device), x3.to(device), x4.to(device), x5.to(device), x6.to(device), y.to(device)
            with torch.no_grad():
                outputs, pooled, y_ = model([x1, x2, x3, x4, x5, x6])
            test_loss += F.cross_entropy(y_, y.squeeze().long()).item()
            pred = y_.max(-1, keepdim=True)[1]
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_probs.extend(torch.softmax(y_, dim=1).cpu().numpy()[:, 1])  # Save predicted probabilities
            if i % 100 == 0:
                print(f'Batch {(batch_idx + 1) * len(x1)}/{len(test_loader.dataset)}, '
                    f'{100. * batch_idx / len(test_loader):.2f}%, Loss: {test_loss:.4f}, '
                    f'Time: {time.time() - start_time:.2f}s')
                start_time = time.time()
    else:
        i = 0
        for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
            i += 1
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            with torch.no_grad():
                outputs, pooled, y_ = model([x1, x2, x3])
            # print(f"y_: {y_.shape}, y: {y.shape}, size: {y.size()}")
            test_loss += F.cross_entropy(y_, y.squeeze().long()).item()
            pred = y_.max(-1, keepdim=True)[1]
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_probs.extend(torch.softmax(y_, dim=1).cpu().numpy()[:, 1])  # Save predicted probabilities
            if i % 100 == 0:
                print(f'Batch {(batch_idx + 1) * len(x1)}/{len(test_loader.dataset)}, '
                    f'{100. * batch_idx / len(test_loader):.2f}%, Loss: {test_loss:.4f}, '
                    f'Time: {time.time() - start_time:.2f}s')
                start_time = time.time()

    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    # Save the confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'],
                yticklabels=['benign', 'malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix_{output_name}.png')

    # Save the ROC curve plot
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{output_name}.png')

    # Save predicted results, original results, and predicted probabilities to a txt file
    results_array = np.column_stack((y_true, y_pred, y_probs))
    header_text = "True label, Predicted label, Predicted Probability"
    np.savetxt(f'results_{output_name}.txt', results_array, fmt='%1.6f', delimiter='\t', header=header_text)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = []  # input char ids
    input_types = []  # segment ids
    input_masks = []  # attention mask
    char_ids = []  # char ids
    start_ids = []  # start ids
    end_ids = []  # end ids
    label = []  # Labels

    BATCH_SIZE = 8

    dataset = "Data/Multiple_dataset/kaggle_multi.csv"
    if IS_CHARBERT:
        test_data_file = "Data/Multiple_dataset/kaggle_multi_charbert.pt"
    else:
        test_data_file = "Data/Multiple_dataset/kaggle_multi_base.pt"

    if os.path.exists(test_data_file):
        test_data = torch.load(test_data_file)
        print("Test data loaded from file.")
    else:
        # 创建反向映射字典 (类别名称 -> 类别ID) 如 {"benign": 0, "malware": 1, "phishing": 2}
        CLASS_NAME_TO_ID = {CLASSES_DICT[i]["name"]: i for i in CLASSES_DICT}

        if IS_CHARBERT:
            char_ids, start_ids, end_ids = dataPreprocessFromCSV(dataset, input_ids, input_types, input_masks, label, IS_CHARBERT, CLASS_NAME_TO_ID)
        else:
            dataPreprocessFromCSV(dataset, input_ids, input_types, input_masks, label, IS_CHARBERT, CLASS_NAME_TO_ID)
        print("load finish")
        # Ensure all input arrays have the same length
        # min_length = min(len(input_ids), len(input_types), len(input_masks), len(label), len(char_ids), len(start_ids), len(end_ids))
        # print(len(input_ids), len(input_types), len(input_masks), len(label), len(char_ids), len(start_ids), len(end_ids))
        # input_ids = input_ids[:min_length]
        # input_types = input_types[:min_length]
        # input_masks = input_masks[:min_length]
        # label = label[:min_length]
        # char_ids = char_ids[:min_length]
        # start_ids = start_ids[:min_length]
        # end_ids = end_ids[:min_length]
        
        # Load data into efficient DataLoaders
        if IS_CHARBERT:
            test_data = TensorDataset(torch.tensor(input_ids).to(DEVICE),
                                    torch.tensor(input_types).to(DEVICE),
                                    torch.tensor(input_masks).to(DEVICE),
                                    torch.tensor(char_ids).to(DEVICE),
                                    torch.tensor(start_ids).to(DEVICE),
                                    torch.tensor(end_ids).to(DEVICE),
                                    torch.tensor(label).to(DEVICE))
        else:
            test_data = TensorDataset(torch.tensor(input_ids).to(DEVICE),
                                    torch.tensor(input_types).to(DEVICE),
                                    torch.tensor(input_masks).to(DEVICE),
                                    torch.tensor(label).to(DEVICE))
        
        # 保存到本地
        torch.save(test_data, test_data_file)
    
    # test_sampler = SequentialSampler(test_data)
    # batch_sampler = BatchSampler(test_sampler, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, drop_last=True)
    
    # Load the pre-trained model

    model_name = "model_epoch_7.pth"

    if IS_CHARBERT:
        model = CharBertModel().to(DEVICE)
    else:
        model = Model().to(DEVICE)

    model.load_state_dict(torch.load(model_name))
    # Test the model
    accuracy, precision, recall, f1 = test_binary(model, DEVICE, test_loader, f'{model_name}_binary')

if __name__ == '__main__':
    main()
