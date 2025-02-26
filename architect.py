import torch
import numpy as np

class Architect:
    """架构优化器，负责更新架构参数"""
    
    def __init__(self, model, lr=3e-4, weight_decay=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.get_alphas(),
            lr=lr, 
            betas=(0.5, 0.999), 
            weight_decay=weight_decay
        )
    
    def step(self, input_valid, target_valid):
        """
        执行架构优化器的一步
        """
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 前向传播
        _, _, logits = self.model(input_valid)
        loss = torch.nn.CrossEntropyLoss()(logits, target_valid)
        
        # 反向传播
        loss.backward()
        
        # 更新架构参数
        self.optimizer.step()