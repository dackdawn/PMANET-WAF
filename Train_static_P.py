import os
import torch
import torch.nn.functional as F
from torch.utils.data import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from data_processing import dataPreprocess_bert, spiltDatast_bert, dataPreprocess_charbert, spiltDatast_charbert
from Model_PMA import Model, CharBertModel
import numpy as np

# batch_size设小
# 显存内存共享

# If IS_CHARBERT is True, use the CharBERT model; otherwise, use the BERT model
IS_CHARBERT = False

CLASS_DICT = {
    0: {"name": "benign", "file": "benign_urls.txt"},
    1: {"name": "malware", "file": "malware_urls.txt"},
    # 2: {"name": "phishing", "file": "phishing_urls.txt"},
    # 3: {"name": "defacement", "file": "defacement_urls.txt"},
    # 可以根据需要添加更多类别
}

NUM_CLASSES = len(CLASS_DICT)

IMBALANCE_CONFIG = {
    "pos_class": 1,  # 正类标签(malware)
    "neg_class": 0,  # 负类标签(benign) 
    "neg_pos_ratio": 3/1,  # 负正样本比例
    "total_pos_samples": 1000  # 将在数据处理时设置
}

log_stream = []

def write_log(log_msg, force=False):
    """
    save log message to file
    """
    log_stream.append(log_msg)
    if len(log_stream) > 100 or force:
        with open('log_want.txt', 'a') as f:
            f.write('\n'.join(log_stream))
        log_stream.clear()


def train(model, device, train_loader, optimizer, epoch):  # Train the model
    """
     Train the model.

    :param model: The model to be trained.
    :param device: The device to run training on (e.g., CPU or GPU).
    :param train_loader: The data loader for training data.
    :param optimizer: The optimization algorithm.
    :param epoch: The current epoch number.
    :return: None
    Change the format:
        for batch_idx, (x1, x2, x3, x4, x5, x6, y) in enumerate(train_loader):
            start_time = time.time()
            x1, x2, x3, x4, x5, x6, y = x1.to(device), x2.to(device), x3.to(device),x4.to(device), x5.to(device), x6.to(device), y.to(device)

            outputs, pooled, y_pred = model([x1, x2, x3, x4, x5, x6])  # Get the prediction results
    """
    model.train()
    best_acc = 0.0

    if IS_CHARBERT:
        for batch_idx, (x1, x2, x3, x4, x5, x6, y) in enumerate(train_loader):
            start_time = time.time()
            x1, x2, x3, x4, x5, x6, y = x1.to(device), x2.to(device), x3.to(device),x4.to(device), x5.to(device), x6.to(device), y.to(device)

            outputs, pooled, y_pred = model([x1, x2, x3, x4, x5, x6])  # Get the prediction results

            # outputs, pooled, y_pred = model([x1, x2, x3])  # Get the prediction results
            model.zero_grad()  # Reset gradients

            # 会报错：RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'
            # loss = F.cross_entropy(y_pred, y.squeeze())  # Calculate the loss
            loss = F.cross_entropy(y_pred.float(), y.squeeze().long())  # Calculate the loss
            loss.backward()

            optimizer.step()
            # if (batch_idx + 1) % 100 == 0:  # Print the loss
            msg = 'Train Epoch: {} [{}/{} ({:.2f}%)]\t Loss: {:.6f}\t Time: {:.2f}s'.format(
                epoch, (batch_idx + 1) * len(x1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), time.time() - start_time)
            print(msg)
            write_log(msg)

    else:
        for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
            start_time = time.time()
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

            outputs, pooled, y_pred = model([x1, x2, x3])  # Get the prediction results
            model.zero_grad()  # Reset gradients

            # 会报错：RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'
            # loss = F.cross_entropy(y_pred, y.squeeze())  # Calculate the loss
            loss = F.cross_entropy(y_pred.float(), y.squeeze().long())  # Calculate the loss
            loss.backward()

            optimizer.step()
            # if (batch_idx + 1) % 100 == 0:  # Print the loss
            msg = 'Train Epoch: {} [{}/{} ({:.2f}%)]\t Loss: {:.6f}\t Time: {:.2f}s'.format(
                epoch, (batch_idx + 1) * len(x1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), time.time() - start_time)
            print(msg)
            write_log(msg)
        
def validation(model, device, test_loader, epoch=0):
    """
    Perform model validation on the test data.

    :param model: The model to be validated.
    :param device: The device to run validation on (e.g., CPU or GPU).
    :param test_loader: The data loader for test data.
    :return: A tuple containing accuracy, precision, recall, and F1 score.
    Change the format:
        for batch_idx, (x1, x2, x3, x4, x5, x6, y) in enumerate(test_loader):
            x1, x2, x3, x4, x5, x6, y = x1.to(device), x2.to(device), x3.to(device),x4.to(device), x5.to(device), x6.to(device), y.to(device)

            with torch.no_grad():
                outputs, pooled, y_ = model([x1, x2, x3, x4, x5, x6])
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    if IS_CHARBERT:
        # charbert的
        for batch_idx, (x1, x2, x3, x4, x5, x6, y) in enumerate(test_loader):
            x1, x2, x3, x4, x5, x6, y = x1.to(device), x2.to(device), x3.to(device),x4.to(device), x5.to(device), x6.to(device), y.to(device)

            with torch.no_grad():
                outputs, pooled, y_ = model([x1, x2, x3, x4, x5, x6])

            # 会报错：RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'
            # test_loss += F.cross_entropy(y_, y.squeeze()).item()
            test_loss += F.cross_entropy(y_.float(), y.squeeze().long()).item()

            pred = y_.max(-1, keepdim=True)[1]  # .max(): 2 outputs, representing the maximum value and its index

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    else:
        # baseline的
        for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

            with torch.no_grad():
                outputs, pooled, y_ = model([x1, x2, x3])

            # 会报错：RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'
            # test_loss += F.cross_entropy(y_, y.squeeze()).item()
            test_loss += F.cross_entropy(y_.float(), y.squeeze().long()).item()

            pred = y_.max(-1, keepdim=True)[1]  # .max(): 2 outputs, representing the maximum value and its index

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 使用CLASS_DICT生成类别标签列表
    class_labels = [CLASS_DICT[i]["name"] for i in range(NUM_CLASSES)]

    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(max(8, NUM_CLASSES), max(6, NUM_CLASSES)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # 保存混淆矩阵并增加旋转标签以提高可读性
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # Save the confusion matrix plot
    plt.savefig(f'confusion_matrix-epoch{epoch}.png')

    log_msg = 'Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100)
    print(log_msg)
    write_log(log_msg)

    # Print detailed confusion matrix information
    print("\nConfusion Matrix Details:")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            true_class = CLASS_DICT[i]["name"]
            pred_class = CLASS_DICT[j]["name"]
            count = cm[i, j]
            print(f"True: {true_class}, Predicted: {pred_class}, Count: {count}")
    
    # Calculate per-class metrics
    print("\nPer-class Metrics:")
    for i in range(NUM_CLASSES):
        class_name = CLASS_DICT[i]["name"]
        true_positives = cm[i, i]
        false_positives = cm[:, i].sum() - true_positives
        false_negatives = cm[i, :].sum() - true_positives
        true_negatives = cm.sum() - (true_positives + false_positives + false_negatives)
        
        class_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        class_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        
        print(f"Class: {class_name}")
        print(f"  Precision: {class_precision:.4f}")
        print(f"  Recall: {class_recall:.4f}")
        print(f"  F1-score: {class_f1:.4f}")
        print(f"  True Positives: {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  False Negatives: {false_negatives}")
        print(f"  True Negatives: {true_negatives}")

    return accuracy, precision, recall, f1

def balance_dataset(input_ids, input_types, input_masks, char_ids, start_ids, end_ids, labels):
    """
    根据固定的正例样本数量和比例平衡数据集
    """
    # 转换为numpy数组便于处理
    labels = np.array(labels)
    
    # 获取正样本和负样本的索引
    pos_indices = np.where(labels == IMBALANCE_CONFIG["pos_class"])[0]
    neg_indices = np.where(labels == IMBALANCE_CONFIG["neg_class"])[0]
    
    # 如果现有正样本数量大于配置的数量，随机采样
    if len(pos_indices) > IMBALANCE_CONFIG["total_pos_samples"]:
        pos_indices = np.random.choice(pos_indices, IMBALANCE_CONFIG["total_pos_samples"], replace=False)
    
    # 根据比例计算需要的负样本数量
    needed_neg_samples = int(IMBALANCE_CONFIG["total_pos_samples"] * IMBALANCE_CONFIG["neg_pos_ratio"])
    
    # 采样负样本
    if len(neg_indices) > needed_neg_samples:
        selected_neg_indices = np.random.choice(neg_indices, needed_neg_samples, replace=False)
    else:
        # 如果负样本不足,使用所有负样本
        selected_neg_indices = neg_indices
        print(f"Warning: Not enough negative samples. Needed {needed_neg_samples}, but only {len(neg_indices)} available.")
    
    # 合并正负样本索引
    selected_indices = np.concatenate([pos_indices, selected_neg_indices])
    
    # 随机打乱索引
    np.random.shuffle(selected_indices)
    
    # 根据选择的索引重构数据集
    balanced_data = {
        'input_ids': [input_ids[i] for i in selected_indices],
        'input_types': [input_types[i] for i in selected_indices],
        'input_masks': [input_masks[i] for i in selected_indices],
        'labels': [labels[i] for i in selected_indices]
    }
    
    if char_ids:
        balanced_data.update({
            'char_ids': [char_ids[i] for i in selected_indices],
            'start_ids': [start_ids[i] for i in selected_indices],
            'end_ids': [end_ids[i] for i in selected_indices]
        })
    
    print(f"Dataset balanced - Total samples: {len(selected_indices)}")
    print(f"Positive samples: {len(pos_indices)}")
    print(f"Selected negative samples: {len(selected_neg_indices)}")
    print(f"Actual ratio: {len(pos_indices)/len(selected_neg_indices):.3f}")
    
    return balanced_data

def main():
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 8
    
    if IS_CHARBERT:
        base_data_file = "_data_charbert.pt"
    else:
        base_data_file = "_data_base.pt"
    
    # 如果任何一个不存在，则需要重新生成数据
    if not os.path.exists(f'train{base_data_file}') or not os.path.exists(f'val{base_data_file}'):
        # 删除pt文件
        if os.path.exists(f'train{base_data_file}'):
            os.remove(f'train{base_data_file}')
        if os.path.exists(f'val{base_data_file}'):
            os.remove(f'val{base_data_file}')

        input_ids = []  # input char ids
        input_types = []  # segment ids
        input_masks = []  # attention mask
        label = []  
        char_ids = []
        start_ids = []
        end_ids = []
        if IS_CHARBERT:
            # dataPreprocess_charbert("benign_urls.txt", input_ids, input_types, input_masks, char_ids, start_ids, end_ids, label, 0)
            # dataPreprocess_charbert("malware_urls.txt", input_ids, input_types, input_masks, char_ids, start_ids, end_ids, label, 1)
            for class_id, class_info in CLASS_DICT.items():
                if os.path.exists(class_info["file"]):
                    dataPreprocess_charbert(class_info["file"], input_ids, input_types, input_masks, 
                                            char_ids, start_ids, end_ids, label, class_id)
                else:
                    print(f"Error: File {class_info['file']} for class {class_info['name']} not found!")
                    exit()
                    
            # 添加数据平衡处理
            balanced_data = balance_dataset(input_ids, input_types, input_masks, 
                                         char_ids, start_ids, end_ids, label)
            print(label)
            # input_ids_train, input_types_train, input_masks_train, char_ids_train, start_ids_train, end_ids_train, y_train, input_ids_val, input_types_val, input_masks_val,char_ids_val,start_ids_val, end_ids_val, y_val = spiltDatast_charbert(input_ids, input_types, input_masks,char_ids,start_ids ,end_ids,label)
            input_ids_train, input_types_train, input_masks_train, char_ids_train, start_ids_train, \
            end_ids_train, y_train, input_ids_val, input_types_val, input_masks_val, char_ids_val, \
            start_ids_val, end_ids_val, y_val = spiltDatast_charbert(
                balanced_data['input_ids'], balanced_data['input_types'], balanced_data['input_masks'],
                balanced_data['char_ids'], balanced_data['start_ids'], balanced_data['end_ids'],
                balanced_data['labels'])
            # print(input_ids_train, input_types_train, input_masks_train, char_ids_train, start_ids_train, end_ids_train, y_train, input_ids_val, input_types_val, input_masks_val,char_ids_val,start_ids_val, end_ids_val, y_val)
        else:
            # dataPreprocess_bert("benign_urls.txt", input_ids, input_types, input_masks, label, 0)
            # dataPreprocess_bert("malware_urls.txt", input_ids, input_types, input_masks, label, 1)
            for class_id, class_info in CLASS_DICT.items():
                if os.path.exists(class_info["file"]):
                    dataPreprocess_bert(class_info["file"], input_ids, input_types, input_masks, label, class_id)
                else:
                    print(f"Error: File {class_info['file']} for class {class_info['name']} not found!")
                    exit()
            # 添加数据平衡处理
            balanced_data = balance_dataset(input_ids, input_types, input_masks, 
                                         None, None, None, label)
            
            # input_ids_train, input_types_train, input_masks_train, y_train, input_ids_val, input_types_val, input_masks_val, y_val = spiltDatast_bert(
            #     input_ids, input_types, input_masks, label
            # )

            # 使用平衡后的数据
            input_ids_train, input_types_train, input_masks_train, y_train, input_ids_val, \
            input_types_val, input_masks_val, y_val = spiltDatast_bert(
                balanced_data['input_ids'], balanced_data['input_types'], 
                balanced_data['input_masks'], balanced_data['labels'])


    # 加载训练数据集
    if os.path.exists(f'train{base_data_file}'):
        train_data = torch.load(f'train{base_data_file}')
        print("train data loaded from file.")
    else:
        # Load data into efficient DataLoaders
        if IS_CHARBERT:
            # charbert版本的
            train_data = TensorDataset(torch.tensor(input_ids_train).to(DEVICE),
                                        torch.tensor(input_types_train).to(DEVICE),
                                        torch.tensor(input_masks_train).to(DEVICE),
                                        torch.tensor(char_ids_train).to(DEVICE),
                                        torch.tensor(start_ids_train).to(DEVICE),
                                        torch.tensor(end_ids_train).to(DEVICE),
                                        torch.tensor(y_train).to(DEVICE))
        elif not IS_CHARBERT:
            train_data = TensorDataset(torch.tensor(input_ids_train).to(DEVICE),
                                        torch.tensor(input_types_train).to(DEVICE),
                                        torch.tensor(input_masks_train).to(DEVICE),
                                        torch.tensor(y_train).to(DEVICE))

        # 保存到本地
        torch.save(train_data, f'train{base_data_file}')

    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, drop_last=True)

    # 加载测试数据集
    if os.path.exists(f'val{base_data_file}'):
        val_data = torch.load(f'val{base_data_file}')
        print("val data loaded from file.")
    else:
        if IS_CHARBERT:
            # charbert版本的
            val_data = TensorDataset(torch.tensor(input_ids_val).to(DEVICE),
                                        torch.tensor(input_types_val).to(DEVICE),
                                        torch.tensor(input_masks_val).to(DEVICE),
                                        torch.tensor(char_ids_val).to(DEVICE),
                                        torch.tensor(start_ids_val).to(DEVICE),
                                        torch.tensor(end_ids_val).to(DEVICE),
                                        torch.tensor(y_val).to(DEVICE))
        elif not IS_CHARBERT:
            val_data = TensorDataset(torch.tensor(input_ids_val).to(DEVICE),
            torch.tensor(input_types_val).to(DEVICE),
            torch.tensor(input_masks_val).to(DEVICE),
            torch.tensor(y_val).to(DEVICE))
        
        # 保存到本地
        torch.save(val_data, f'val{base_data_file}')
        
    val_sampler = SequentialSampler(val_data)
    val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE, drop_last=True)
    
    if IS_CHARBERT:
        model = CharBertModel(num_classes=NUM_CLASSES)
    elif not IS_CHARBERT:
        model = Model()
    
    # 在创建模型后添加 DataParallel 封装
    # if torch.cuda.device_count() > 1:
    #     print("发现 {} 张GPU，使用DataParallel进行多卡训练".format(torch.cuda.device_count()))
    #     model = torch.nn.DataParallel(model)

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    best_acc = 0.0
    NUM_EPOCHS = 2
    for epoch in range(1, NUM_EPOCHS + 1):  # 3 epochs
        train(model, DEVICE, train_loader, optimizer, epoch)
        acc, precision, recall, f1 = validation(model, DEVICE, val_loader, epoch)
        if best_acc < acc:
            best_acc = acc
            
        if IS_CHARBERT:
            PATH = f'./charbert_model_staticP({IMBALANCE_CONFIG["neg_pos_ratio"]})_epoch_{epoch}.pth'
        else:
            PATH = f'./bert_model_staticP({IMBALANCE_CONFIG["neg_pos_ratio"]})_epoch_{epoch}.pth'
        torch.save(model.state_dict(), PATH)  # Save the best model
        print("acc is: {:.4f}, best acc is {:.4f}n".format(acc, best_acc))

if __name__ == '__main__':
    main()
    write_log("Training finished.", True)
