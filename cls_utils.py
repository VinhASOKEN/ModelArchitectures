import os
import torch

"""
Xây dựng hàm loss
"""
def intit_loss():
    loss = torch.nn.CrossEntropyLoss()
    return loss


"""
Xây dựng hàm Optimizer
"""
def init_optimizeer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return optimizer


"""
Xây dựng hàm scheduler
"""
def init_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    return scheduler