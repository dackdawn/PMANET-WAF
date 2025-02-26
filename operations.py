import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import CBAMLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 特征融合操作
class FuseOps:
    @staticmethod
    def conv1d_fuse(x1, x2, hidden_size):
        # 当前实现方式: 拼接 + 1D卷积
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)
        
        # 拼接两个张量
        x = torch.cat([x1, x2], dim=-1)  # [batch_size, seq_len, hidden_size*2]
        
        # 重塑张量以适应卷积层
        x = x.transpose(1, 2)  # [batch_size, hidden_size*2, seq_len]
        
        # 创建1D卷积层
        fuse_layer = nn.Conv1d(2 * hidden_size, hidden_size, kernel_size=1).to(DEVICE)
        
        # 应用卷积融合
        y = fuse_layer(x)  # [batch_size, hidden_size, seq_len]
        
        # 重塑回原始形状
        y_output = y.transpose(1, 2)  # [batch_size, seq_len, hidden_size]
        
        return y_output

    @staticmethod
    def attention_fuse(x1, x2, hidden_size):
        # 注意力加权融合
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)
        
        # 创建查询、键和值
        q = nn.Linear(hidden_size, hidden_size).to(DEVICE)(x1)
        k = nn.Linear(hidden_size, hidden_size).to(DEVICE)(x2)
        v = nn.Linear(hidden_size, hidden_size).to(DEVICE)(x2)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-1, -2)) / (hidden_size ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权和
        context = torch.matmul(attn_weights, v)
        
        # 残差连接
        output = x1 + context
        
        return output

    @staticmethod
    def gated_fuse(x1, x2, hidden_size):
        # 门控融合机制
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)
        
        # 创建门控网络
        gate = nn.Linear(hidden_size * 2, hidden_size).to(DEVICE)
        
        # 拼接输入
        concat = torch.cat([x1, x2], dim=-1)
        
        # 计算门控权重
        z = torch.sigmoid(gate(concat))
        
        # 应用门控
        output = z * x1 + (1 - z) * x2
        
        return output

# 2. 注意力机制
class AttentionOps:
    @staticmethod
    def cbam_attention(x, channel):
        # 当前实现: CBAM注意力
        model_cbam = CBAMLayer(channel=channel).to(DEVICE)
        return model_cbam(x)
    
    @staticmethod
    def self_attention(x, channel):
        # 自注意力机制
        batch_size, channels, seq_len, hidden_size = x.size()
        
        # 重塑为自注意力格式 [batch_size*channels, seq_len, hidden_size]
        x_reshaped = x.view(batch_size * channels, seq_len, hidden_size)
        
        # 自注意力层
        q = nn.Linear(hidden_size, hidden_size).to(DEVICE)(x_reshaped)
        k = nn.Linear(hidden_size, hidden_size).to(DEVICE)(x_reshaped)
        v = nn.Linear(hidden_size, hidden_size).to(DEVICE)(x_reshaped)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-1, -2)) / (hidden_size ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # 重塑回原始形状
        output = context.view(batch_size, channels, seq_len, hidden_size)
        
        return output
    
    @staticmethod
    def se_attention(x, channel):
        # SE注意力块
        batch_size, channels, seq_len, hidden_size = x.size()
        
        # 全局平均池化
        squeeze = torch.mean(x, dim=(2, 3), keepdim=True)  # [batch_size, channels, 1, 1]
        
        # 两个FC层
        fc1 = nn.Linear(channels, channels // 4).to(DEVICE)
        fc2 = nn.Linear(channels // 4, channels).to(DEVICE)
        
        # 激活函数
        excitation = fc2(F.relu(fc1(squeeze.squeeze(-1).squeeze(-1)))).unsqueeze(-1).unsqueeze(-1)
        excitation = torch.sigmoid(excitation)  # [batch_size, channels, 1, 1]
        
        # 缩放
        output = x * excitation
        
        return output

    @staticmethod
    def no_attention(x, channel):
        # 跳过连接（无注意力）
        return x

# 3. 空间金字塔池化策略
class PyramidPoolingOps:
    @staticmethod
    def pyramid_pooling(x, levels):
        # 当前实现: 空间金字塔池化
        pooled_features = []
        
        for level in levels:
            # 计算每个级别的池化窗口大小
            window_size = x.size(1) // level
            
            # 使用平均池化
            pooled = F.avg_pool2d(x.permute(0, 3, 2, 1), (1, window_size)).permute(0, 3, 2, 1)
            
            # 添加池化结果
            pooled_features.append(pooled)
        
        # 拼接池化结果
        return torch.cat(pooled_features, dim=1)