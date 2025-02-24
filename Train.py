import torch
import torch.nn.functional as F
from torch.utils.data import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from data_processing import dataPreprocess_bert, spiltDatast_bert, dataPreprocess_charbert, spiltDatast_charbert
from Model_PMA import Model, CharBertModel

# batch_size设小
# 显存内存共享

# If IS_CHARBERT is True, use the CharBERT model; otherwise, use the BERT model
IS_CHARBERT = True


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
            msg = 'Train Epoch: {} [{}/{} ({:.2f}%)]/t Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                                len(train_loader.dataset),
                                                                                100. * batch_idx / len(train_loader),
                                                                                loss.item())  # Remember to use loss.item()
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
            msg = 'Train Epoch: {} [{}/{} ({:.2f}%)]/t Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                                len(train_loader.dataset),
                                                                                100. * batch_idx / len(train_loader),
                                                                                loss.item())  # Remember to use loss.item()
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
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'],
                yticklabels=['benign', 'malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the confusion matrix plot
    plt.savefig(f'confusion_matrix-epoch{epoch}.png')

    log_msg = 'Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100)
    print(log_msg)
    write_log(log_msg)

    return accuracy, precision, recall, f1


def main():
    input_ids = []  # input char ids
    input_types = []  # segment ids
    input_masks = []  # attention mask
    label = []  
    char_ids = []
    start_ids = []
    end_ids = []
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if IS_CHARBERT:
        dataPreprocess_charbert("benign_urls.txt", input_ids, input_types, input_masks, char_ids, start_ids, end_ids, label, 0)
        dataPreprocess_charbert("malware_urls.txt", input_ids, input_types, input_masks, char_ids, start_ids, end_ids, label, 1)
        input_ids_train, input_types_train, input_masks_train, char_ids_train, start_ids_train, end_ids_train, y_train, input_ids_val, input_types_val, input_masks_val,char_ids_val,start_ids_val, end_ids_val, y_val = spiltDatast_charbert(
                input_ids, input_types, input_masks,char_ids,start_ids ,end_ids,label)
        print(input_ids_train, input_types_train, input_masks_train, char_ids_train, start_ids_train, end_ids_train, y_train, input_ids_val, input_types_val, input_masks_val,char_ids_val,start_ids_val, end_ids_val, y_val)
    elif not IS_CHARBERT:
        dataPreprocess_bert("benign_urls.txt", input_ids, input_types, input_masks, label, 0)
        dataPreprocess_bert("malware_urls.txt", input_ids, input_types, input_masks, label, 1)
        input_ids_train, input_types_train, input_masks_train, y_train, input_ids_val, input_types_val, input_masks_val, y_val = spiltDatast_bert(
            input_ids, input_types, input_masks, label
        )

    """
       input_ids_train, input_types_train, input_masks_train, char_ids_train, start_ids_train, end_ids_train, y_train, input_ids_val, input_types_val, input_masks_val,char_ids_val,start_ids_val, end_ids_val, y_val = spiltDatast_charbert(
            input_ids, input_types, input_masks,char_ids,start_ids ,end_ids,label)
    """
    # Load data into efficient DataLoaders
    BATCH_SIZE = 8
           
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

    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

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
    
    val_sampler = SequentialSampler(val_data)
    val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)
    
    if IS_CHARBERT:
        model = CharBertModel()
    elif not IS_CHARBERT:
        model = Model()
    
    # 在创建模型后添加 DataParallel 封装
    # if torch.cuda.device_count() > 1:
    #     print("发现 {} 张GPU，使用DataParallel进行多卡训练".format(torch.cuda.device_count()))
    #     model = torch.nn.DataParallel(model)

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    best_acc = 0.0
    NUM_EPOCHS = 3
    for epoch in range(1, NUM_EPOCHS + 1):  # 3 epochs
        train(model, DEVICE, train_loader, optimizer, epoch)
        acc, precision, recall, f1 = validation(model, DEVICE, val_loader, epoch)
        if best_acc < acc:
            best_acc = acc
            
        if IS_CHARBERT:
            PATH = f'./charbert_model_epoch_{epoch}.pth'
        else:
            PATH = f'./bert_model_epoch_{epoch}.pth'
        torch.save(model.state_dict(), PATH)  # Save the best model
        print("acc is: {:.4f}, best acc is {:.4f}n".format(acc, best_acc))

if __name__ == '__main__':
    main()
