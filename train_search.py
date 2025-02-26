import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
from model_search import SuperCharBertModel
from architect import Architect
from data_processing import dataPreprocess_charbert, spiltDatast_charbert

# 设备设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 超参数
BATCH_SIZE = 16
SEARCH_EPOCHS = 50
LEARNING_RATE = 2e-5
ARCH_LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
ARCH_WEIGHT_DECAY = 1e-3
NUM_CLASSES = 2

# 定义类别字典
CLASS_DICT = {
    0: {"name": "benign", "file": "benign_urls.txt"},
    1: {"name": "malware", "file": "malware_urls.txt"},
    2: {"name": "phishing", "file": "phishing_urls.txt"},
    # 3: {"name": "defacement", "file": "defacement_urls.txt"},
    # 可以根据需要添加更多类别
}

NUM_CLASSES = len(CLASS_DICT)

def train_search(train_loader, valid_loader, model, architect, optimizer, epoch):
    model.train()
    
    for step, (train_batch, valid_batch) in enumerate(zip(train_loader, valid_loader)):
        # 训练批次数据
        train_inputs = [item.to(DEVICE) for item in train_batch[:-1]]
        train_labels = train_batch[-1].to(DEVICE)
        
        # 验证批次数据
        valid_inputs = [item.to(DEVICE) for item in valid_batch[:-1]]
        valid_labels = valid_batch[-1].to(DEVICE)
        
        # 1. 更新架构参数 (on validation data)
        architect.step(valid_inputs, valid_labels)
        
        # 2. 更新网络权重 (on training data)
        optimizer.zero_grad()
        _, _, logits = model(train_inputs)
        loss = nn.CrossEntropyLoss()(logits, train_labels)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            pred = logits.argmax(dim=1, keepdim=True)
            correct = pred.eq(train_labels.view_as(pred)).sum().item()
            accuracy = correct / train_labels.size(0)
            
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
            model.show_arch_parameters()

def validation(valid_loader, model, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in valid_loader:
            inputs = [item.to(DEVICE) for item in batch[:-1]]
            labels = batch[-1].to(DEVICE)
            
            _, _, logits = model(inputs)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            val_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"Validation - Epoch: {epoch}, Loss: {val_loss/len(valid_loader):.4f}, Acc: {accuracy:.4f}")
    return accuracy

def main():
    # 加载数据
    print("Loading and preprocessing data...")
    
    # 数据文件
    base_data_file = "_search_data.pt"
    
    input_ids, input_types, input_masks, label = [], [], [], []
    char_ids, start_ids, end_ids = [], [], []
    

    # 加载和处理数据集
    if os.path.exists(f'train{base_data_file}') and os.path.exists(f'val{base_data_file}'):
        print("Loading preprocessed data from files...")
        train_data = torch.load(f'train{base_data_file}')
        val_data = torch.load(f'val{base_data_file}')
    else:
        print("Processing raw data...")
        for class_id, class_info in CLASS_DICT.items():
            if os.path.exists(class_info["file"]):
                dataPreprocess_charbert(class_info["file"], input_ids, input_types, input_masks, 
                                        char_ids, start_ids, end_ids, label, class_id)
                
                (
                    input_ids_train,
                    input_types_train,
                    input_masks_train,
                    char_ids_train,
                    start_ids_train,
                    end_ids_train,
                    y_train,
                    input_ids_val,
                    input_types_val,
                    input_masks_val,
                    char_ids_val,
                    start_ids_val,
                    end_ids_val,
                    y_val
                ) = spiltDatast_charbert(
                    input_ids, 
                    input_types, 
                    input_masks, 
                    char_ids, 
                    start_ids, 
                    end_ids, 
                    label
                )

            else:
                print(f"Error: File {class_info['file']} for class {class_info['name']} not found!")
                exit()
        print(label)

        # 加载训练数据集
        if os.path.exists(f'train{base_data_file}'):
            train_data = torch.load(f'train{base_data_file}')
            print("train data loaded from file.")
        else:
            # Load data into efficient DataLoaders
            train_data = TensorDataset(torch.tensor(input_ids_train).to(DEVICE),
                                        torch.tensor(input_types_train).to(DEVICE),
                                        torch.tensor(input_masks_train).to(DEVICE),
                                        torch.tensor(char_ids_train).to(DEVICE),
                                        torch.tensor(start_ids_train).to(DEVICE),
                                        torch.tensor(end_ids_train).to(DEVICE),
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
            val_data = TensorDataset(torch.tensor(input_ids_val).to(DEVICE),
                                        torch.tensor(input_types_val).to(DEVICE),
                                        torch.tensor(input_masks_val).to(DEVICE),
                                        torch.tensor(char_ids_val).to(DEVICE),
                                        torch.tensor(start_ids_val).to(DEVICE),
                                        torch.tensor(end_ids_val).to(DEVICE),
                                        torch.tensor(y_val).to(DEVICE))
            
            # 保存到本地
            torch.save(val_data, f'val{base_data_file}')
            
        val_sampler = SequentialSampler(val_data)
        val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE, drop_last=True)

    # 初始化超网模型
    model = SuperCharBertModel(num_classes=NUM_CLASSES).to(DEVICE)
    
    # 初始化架构优化器
    architect = Architect(model, lr=ARCH_LEARNING_RATE, weight_decay=ARCH_WEIGHT_DECAY)
    
    # 初始化模型权重优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=SEARCH_EPOCHS
    )
    
    # 记录最佳验证准确率和对应的架构
    best_acc = 0.0
    best_arch = None
    
    # 训练循环
    for epoch in range(SEARCH_EPOCHS):
        print(f"\n{'='*20} Epoch {epoch+1}/{SEARCH_EPOCHS} {'='*20}")
        
        # 训练一个周期
        train_search(train_loader, val_loader, model, architect, optimizer, epoch)
        
        # 在验证集上评估
        val_acc = validation(val_loader, model, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型和架构
        if val_acc > best_acc:
            best_acc = val_acc
            best_arch = {
                'layer_weights': model.layer_weights.data.cpu().numpy().tolist(),
                'hidden_size_weights': model.hidden_size_weights.data.cpu().numpy().tolist(),
                'attention_type_weights': model.attention_type_weights.data.cpu().numpy().tolist(),
                'pyramid_level_weights': model.pyramid_level_weights.data.cpu().numpy().tolist()
            }
            
            # 保存模型和架构参数
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'architect_state_dict': architect.optimizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'best_arch': best_arch
            }, f'search_checkpoint_epoch_{epoch+1}.pth')
            
            print(f"New best architecture found! Validation accuracy: {val_acc:.4f}")
            print("Best architecture:")
            model.show_arch_parameters()
    
    # 打印最终架构
    print("\n" + "="*50)
    print(f"Search completed. Best validation accuracy: {best_acc:.4f}")
    print("Best architecture found:")
    for k, v in best_arch.items():
        print(f"{k}: {v}")
    
    # 保存最终最佳架构
    import json
    with open('best_architecture.json', 'w') as f:
        json.dump(best_arch, f, indent=2)
    
    print("Best architecture saved to 'best_architecture.json'")

if __name__ == "__main__":
    main()