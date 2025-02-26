import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import FuseOps, AttentionOps, PyramidPoolingOps
from Model_CharBERT import CharBERTModel
from transformers import BertConfig
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MixedOp(nn.Module):
    """混合操作类，用于权重化组合多个候选操作"""
    
    def __init__(self, ops):
        super(MixedOp, self).__init__()
        self.ops = nn.ModuleList(ops)
        self.weights = nn.Parameter(torch.ones(len(ops)) / len(ops))
        
    def forward(self, *args):
        # 使用softmax获取正规化的权重
        weights = F.softmax(self.weights, dim=0)
        
        # 对每个操作应用权重并求和
        return sum(w * op(*args) for w, op in zip(weights, self.ops))

class SuperCharBertModel(nn.Module):
    """超网模型，包含所有可能的架构选择"""
    
    def __init__(self, num_classes=2):
        super(SuperCharBertModel, self).__init__()
        
        # 加载CharBERT基础模型
        config = BertConfig.from_pretrained('charbert-bert-wiki')
        self.bert = CharBERTModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # 基本组件
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(768, self.num_classes)
        
        # 1. 特征融合层配置


        # 1.1 隐藏层大小选项
        self.hidden_size_options = [256, 512, 768]
        self.hidden_size_weights = nn.Parameter(torch.ones(len(self.hidden_size_options)) / len(self.hidden_size_options))
        
        # 为每个隐藏层大小创建融合层
        self.fuse_layers = nn.ModuleList([
            nn.Conv1d(2 * hidden_size, hidden_size, kernel_size=1).to(DEVICE)
            for hidden_size in self.hidden_size_options
        ])
        
        # 1.2 特征选择层配置
        self.layer_choices = [
            list(range(12)),                # 全部层
            list(range(6)),                 # 浅层
            list(range(6, 12)),             # 深层
            # [0, 1, 10, 11],                 # 首尾层
            # list(range(0, 12, 2)),          # 间隔选择
            # list(range(0, 12, 2)),          # 奇数层
            # list(range(1, 12, 2)),          # 偶数层
            # list(range(3, 9)),              # 中间层
        ]
        self.layer_choice_weights = nn.Parameter(torch.ones(len(self.layer_choices)) / len(self.layer_choices))
        
        # 2. 注意力机制选择
        self.attention_ops = [
            lambda x: AttentionOps.cbam_attention(x, channel=12),
            # lambda x: AttentionOps.self_attention(x, channel=12),
            # lambda x: AttentionOps.se_attention(x, channel=12),
            lambda x: AttentionOps.no_attention(x, channel=12)
        ]
        self.attention_weights = nn.Parameter(torch.ones(len(self.attention_ops)) / len(self.attention_ops))
        
        # 3. 池化策略选择
        self.pooling_configs = [
            [1, 2, 3, 4],      # 当前配置
            [1, 2, 4, 6],      # 更宽分布
            # [1, 2, 3, 6],      # 更密集分布
            # [3, 6, 9, 12],     # 均匀分布
            # [1, 4, 8],         # 更少级别
            # [1, 2, 3, 4, 6],   # 更多级别
            # [1, 3, 9]          # 非线性分布
        ]
        self.pooling_weights = nn.Parameter(torch.ones(len(self.pooling_configs)) / len(self.pooling_configs))
        
    def forward(self, x):
        # 1. 输入处理
        context = x[0]      # 输入token ids
        types = x[1]        # token类型ids
        mask = x[2]         # 注意力mask
        char_ids = x[3]     # 字符ids
        start_ids = x[4]    # 开始位置
        end_ids = x[5]      # 结束位置
        
        # 2. 通过CharBERT获取词级和字符级的隐藏状态
        all_hidden_states_word, all_hidden_states_char, pooled_output = self.bert(
            char_input_ids=char_ids,
            start_ids=start_ids,
            end_ids=end_ids,
            input_ids=context,
            attention_mask=mask,
            token_type_ids=types
        )
        
        # 3. 特征层选择 - 使用softmax获取正规化的权重
        layer_weights = F.softmax(self.layer_choice_weights, dim=0)
        
        # 4. 特征融合 - 根据选择的层和融合方法
        hidden_size_weights = F.softmax(self.hidden_size_weights, dim=0)
        
        fuse_outputs = []
        # 对每个层组合应用融合
        for i, weight in enumerate(layer_weights):
            if weight < 1e-3:  # 忽略权重很小的层组合
                continue
                
            selected_layers = self.layer_choices[i]
            
            layer_outputs = []
            for layer_idx in selected_layers:
                if layer_idx >= len(all_hidden_states_word):
                    continue
                    
                x1 = all_hidden_states_word[layer_idx].to(DEVICE)
                x2 = all_hidden_states_char[layer_idx].to(DEVICE)
                
                # 拼接两个张量
                x_concat = torch.cat([x1, x2], dim=-1)  # [batch_size, seq_len, hidden_size*2]
                x_concat = x_concat.transpose(1, 2)  # [batch_size, hidden_size*2, seq_len]
                
                # 对每个隐藏层大小应用融合
                layer_output = 0
                for j, h_weight in enumerate(hidden_size_weights):
                    if h_weight < 1e-3:  # 忽略权重很小的隐藏层大小
                        continue
                        
                    # 如果隐藏层大小不是768，需要调整大小
                    if self.hidden_size_options[j] != 768:
                        # 重塑为2D以便调整大小
                        batch_size, channels, seq_len = x_concat.size()
                        x_reshaped = x_concat.view(batch_size, channels, -1)
                        
                        # 调整通道数
                        if self.hidden_size_options[j] < 768 * 2:
                            # 减少通道
                            x_resized = x_reshaped[:, :self.hidden_size_options[j]*2, :]
                        else:
                            # 增加通道（填充）
                            pad_size = self.hidden_size_options[j]*2 - channels
                            x_resized = F.pad(x_reshaped, (0, 0, 0, pad_size))
                        
                        # 重塑回3D
                        x_resized = x_resized.view(batch_size, self.hidden_size_options[j]*2, seq_len)
                    else:
                        x_resized = x_concat
                    
                    y = self.fuse_layers[j](x_resized)
                    y_output = y.transpose(1, 2)  # [batch_size, seq_len, hidden_size_options[j]]
                    
                    # 如果隐藏层大小不是768，调整回768
                    if self.hidden_size_options[j] != 768:
                        y_output = nn.Linear(self.hidden_size_options[j], 768).to(DEVICE)(y_output)
                    
                    layer_output = layer_output + h_weight * y_output
                
                layer_outputs.append(layer_output)
            
            # 平均所有选定层的输出
            if layer_outputs:
                avg_output = sum(layer_outputs) / len(layer_outputs)
                fuse_outputs.append(weight * avg_output)
        
        # 结合所有融合输出
        fuse_output = sum(fuse_outputs)
        
        # 5. 构建金字塔
        pyramid = torch.stack([fuse_output] * 12, dim=0).permute(1, 0, 2, 3)  # [batch_size, 12, seq_len, 768]
        
        # 6. 应用注意力机制 - 使用softmax获取正规化的权重
        attention_weights = F.softmax(self.attention_weights, dim=0)
        pos_pooled = sum(w * op(pyramid) for w, op in zip(attention_weights, self.attention_ops))
        
        # 7. 应用金字塔池化 - 使用softmax获取正规化的权重
        pooling_weights = F.softmax(self.pooling_weights, dim=0)
        
        # 对每个池化配置应用池化
        pooled_results = []
        for i, weight in enumerate(pooling_weights):
            if weight < 1e-3:  # 忽略权重很小的池化配置
                continue
                
            # 应用当前池化配置
            levels = self.pooling_configs[i]
            pooled_features = []
            
            for level in levels:
                # 计算每个级别的池化窗口大小
                window_size = max(pos_pooled.size(1) // level, 1)
                
                # 使用平均池化
                pooled = F.avg_pool2d(pos_pooled.permute(0, 3, 2, 1), (1, window_size)).permute(0, 3, 2, 1)
                
                # 添加池化结果
                pooled_features.append(pooled)
            
            # 拼接当前配置的池化结果
            if pooled_features:
                concatenated = torch.cat(pooled_features, dim=1)
                pooled_results.append(weight * concatenated)
        
        # 结合所有池化结果
        concatenated_features = sum(pooled_results)
        
        # 8. 压缩特征
        compressed_feature_tensor = torch.mean(concatenated_features, dim=2)
        compressed_feature_tensor = torch.mean(compressed_feature_tensor, dim=1)
        
        # 9. 分类
        out = self.dropout(compressed_feature_tensor)
        out = self.fc(out)
        
        return pyramid, pooled_output, out
    
    def get_alphas(self):
        """获取所有可学习的架构参数"""
        return [
            self.layer_choice_weights,
            self.hidden_size_weights,
            self.attention_weights,
            self.pooling_weights
        ]
    
    def show_arch_parameters(self):
        """显示架构参数"""
        print("层选择权重:", F.softmax(self.layer_choice_weights, dim=0))
        print("隐藏层大小权重:", F.softmax(self.hidden_size_weights, dim=0))
        print("注意力机制权重:", F.softmax(self.attention_weights, dim=0))
        print("池化配置权重:", F.softmax(self.pooling_weights, dim=0))
    
    def get_best_arch(self):
        """获取最佳架构配置"""
        layer_idx = torch.argmax(F.softmax(self.layer_choice_weights, dim=0)).item()
        hidden_size_idx = torch.argmax(F.softmax(self.hidden_size_weights, dim=0)).item()
        attention_idx = torch.argmax(F.softmax(self.attention_weights, dim=0)).item()
        pooling_idx = torch.argmax(F.softmax(self.pooling_weights, dim=0)).item()
        
        attention_types = [
            "CBAM",
            # "Self-Attention",
            # "SE",
            "No Attention",
        ]
        
        return {
            "selected_layers": self.layer_choices[layer_idx],
            "hidden_size": self.hidden_size_options[hidden_size_idx],
            "attention_type": attention_types[attention_idx],
            "pooling_levels": self.pooling_configs[pooling_idx]
        }