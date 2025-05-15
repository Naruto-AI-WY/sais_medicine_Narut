#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import random
import pandas as pd
import time
import sys
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split

# 导入工具类
from run import RNAGraphBuilder, Config

# 设置日志函数
def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")
    sys.stdout.flush()  # 立即刷新输出

# 为训练创建一个简化版本的GNN模型
class TrainGNNModel(torch.nn.Module):
    def __init__(self, input_dim=31):
        super().__init__()
        log(f"初始化模型，输入维度: {input_dim}")
        # 特征编码
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, Config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.dropout)
        )

        # GNN层
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(Config.hidden_dim, Config.hidden_dim)
        self.conv2 = GCNConv(Config.hidden_dim, Config.hidden_dim)

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(Config.hidden_dim, len(Config.seq_vocab))
        )
        log("模型初始化完成")

    def forward(self, data):
        # 节点特征编码
        x = self.encoder(data.x)  # [N, hidden]

        # 图卷积
        edge_index = data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        # 节点分类
        logits = self.cls_head(x)  # [N, 4]
        return logits

class RNADataset(Dataset):
    """RNA骨架坐标与序列数据集"""
    def __init__(self, coords_dir, seqs_dir, transform=None, max_samples=None):
        """初始化数据集"""
        self.coords_dir = coords_dir
        self.seqs_dir = seqs_dir
        self.transform = transform
        
        # 获取所有坐标文件
        self.coord_files = sorted(glob.glob(os.path.join(coords_dir, "*.npy")))
        if max_samples and max_samples > 0:
            self.coord_files = self.coord_files[:max_samples]
        log(f"找到 {len(self.coord_files)} 个坐标文件")
        
        # 构建坐标文件对应的序列文件映射
        self.seq_files = {}
        for seq_file in glob.glob(os.path.join(seqs_dir, "*.fasta")):
            base_name = os.path.basename(seq_file).split('.')[0]
            self.seq_files[base_name] = seq_file
            
        # 确认每个坐标都有对应的序列
        self.valid_samples = []
        for coord_file in self.coord_files:
            base_name = os.path.basename(coord_file).split('.')[0]
            if base_name in self.seq_files:
                self.valid_samples.append((coord_file, self.seq_files[base_name]))
        
        log(f"有效样本: {len(self.valid_samples)}/{len(self.coord_files)}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        try:
            coord_file, seq_file = self.valid_samples[idx]
            
            # 读取骨架坐标
            log(f"加载坐标文件: {coord_file}") if idx < 3 else None
            coord = np.load(coord_file)
            
            # 读取序列
            with open(seq_file, "r") as f:
                lines = f.readlines()
                # 剔除FASTA文件的头部（以>开始的行）
                seq = "".join([line.strip() for line in lines if not line.startswith(">")])
            
            # 确保坐标和序列长度匹配
            if coord.shape[0] != len(seq):
                log(f"警告: {coord_file}坐标长度({coord.shape[0]})与序列长度({len(seq)})不匹配!")
                # 取最短的长度
                min_len = min(coord.shape[0], len(seq))
                coord = coord[:min_len]
                seq = seq[:min_len]
            
            # 数据预处理
            coord = np.nan_to_num(coord, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 应用自定义转换
            if self.transform:
                coord, seq = self.transform(coord, seq)
                
            # 构建图数据
            graph = RNAGraphBuilder.build_graph(coord, seq)
            log(f"样本 {idx} 处理完成，图节点数: {graph.num_nodes}") if idx < 3 else None
            
            return graph
        except Exception as e:
            log(f"处理样本 {idx} 时出错: {str(e)}")
            # 创建一个简单的替代图
            dummy_coord = np.zeros((10, 7, 3))
            dummy_seq = "A" * 10
            return RNAGraphBuilder.build_graph(dummy_coord, dummy_seq)

def main():
    # 设置随机种子
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    random.seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.seed)
    
    # 确定设备
    device = torch.device(Config.device)
    log(f"使用设备: {device}")
    
    # 数据目录
    coords_dir = "./RNA_design_public/RNAdesignv1/train/coords"
    seqs_dir = "./RNA_design_public/RNAdesignv1/train/seqs"
    
    # 创建一个小型数据集用于调试 - 减少样本数
    log("加载数据集 (限制为100个样本)...")
    dataset = RNADataset(coords_dir, seqs_dir, transform=None, max_samples=100)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    log(f"训练集大小: {len(train_dataset)}")
    log(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器 - 使用小批量和单线程加载
    batch_size = 8  # 减小批量大小
    log(f"创建DataLoader，批量大小: {batch_size}, 单线程加载")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    try:
        # 获取一个样本来确定输入维度
        log("获取样本以确定输入维度...")
        sample_data = dataset[0]
        input_dim = sample_data.x.size(1)
        log(f"检测到输入维度: {input_dim}")
        
        # 创建模型
        log("创建模型...")
        model = TrainGNNModel(input_dim=input_dim).to(device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=1e-4)
        
        # 训练一个epoch
        log("开始训练...")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, data in enumerate(train_loader):
            log(f"处理批次 {batch_idx+1}/{len(train_loader)}")
            
            # 将数据移到指定设备
            data = data.to(device)
            
            # 清除之前的梯度
            optimizer.zero_grad()
            
            try:
                # 前向传播
                log(f"批次 {batch_idx+1}: 前向传播...")
                outputs = model(data)
                
                # 计算损失
                log(f"批次 {batch_idx+1}: 计算损失...")
                loss = criterion(outputs, data.y)
                
                # 反向传播
                log(f"批次 {batch_idx+1}: 反向传播...")
                loss.backward()
                
                # 更新参数
                log(f"批次 {batch_idx+1}: 参数更新...")
                optimizer.step()
                
                # 统计
                train_loss += loss.item()
                preds = outputs.argmax(dim=1)
                train_correct += (preds == data.y).sum().item()
                train_total += data.y.size(0)
                
                log(f"批次 {batch_idx+1} 完成: 损失={loss.item():.4f}, 准确率={train_correct/train_total:.4f}")
                
                # 只训练5个批次，用于调试
                if batch_idx >= 4:
                    log("已完成5个批次，提前停止训练")
                    break
            except Exception as e:
                log(f"批次 {batch_idx+1} 处理出错: {str(e)}")
                continue
        
        # 简单评估
        log("\n训练完成，开始评估...")
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                log(f"评估批次 {batch_idx+1}/{len(val_loader)}")
                data = data.to(device)
                outputs = model(data)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == data.y).sum().item()
                val_total += data.y.size(0)
                
                # 只评估2个批次
                if batch_idx >= 1:
                    break
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        log(f"验证集准确率: {val_acc:.4f}")
        
        # 保存模型
        log("保存模型为 debug_model.pth")
        torch.save(model.state_dict(), "debug_model.pth")
        
        log("调试训练完成")
        
    except Exception as e:
        log(f"训练过程出错: {str(e)}")
        import traceback
        log(traceback.format_exc())

if __name__ == "__main__":
    log("开始执行脚本")
    main()
