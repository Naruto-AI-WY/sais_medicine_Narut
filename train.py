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
import matplotlib.pyplot as plt
import time
import sys
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader  # 使用PyTorch Geometric的DataLoader
from torch_geometric.data import Batch  # 用于批处理图数据
from sklearn.model_selection import train_test_split

# 导入我们的工具类
from run import RNAGraphBuilder, Config

# 日志目录
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 检查点目录
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 设置日志函数
def log(message, filename="train.log"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    sys.stdout.flush()  # 立即刷新输出
    
    # 同时写入日志文件
    with open(os.path.join(LOG_DIR, filename), "a") as f:
        f.write(log_message + "\n")

# 为训练创建一个更稳健的GNN模型
class TrainGNNModel(torch.nn.Module):
    def __init__(self, input_dim=31):  # 默认输入维度设为31，与实际数据匹配
        super().__init__()
        log(f"初始化GNN模型，输入维度: {input_dim}, 隐藏维度: {Config.hidden_dim}")
        
        # 设置dropout值
        dropout_rate = getattr(Config, 'dropout', 0.3)
        
        # 特征编码
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, Config.hidden_dim),
            nn.BatchNorm1d(Config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 导入图神经网络层
        from torch_geometric.nn import GCNConv, GATConv, ResGatedGraphConv
        
        # GNN层
        self.conv_layers = nn.ModuleList([
            GCNConv(Config.hidden_dim, Config.hidden_dim),
            GATConv(Config.hidden_dim, Config.hidden_dim, heads=1),  # 使用注意力机制，单头以减少计算负担
            ResGatedGraphConv(Config.hidden_dim, Config.hidden_dim)  # 使用门控机制
        ])
        
        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(Config.hidden_dim) for _ in range(len(self.conv_layers))
        ])
        
        # 通用dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类头 - 更简单的结构以减少过拟合风险
        self.cls_head = nn.Sequential(
            nn.Linear(Config.hidden_dim, Config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(Config.hidden_dim // 2, len(Config.seq_vocab))
        )
        
        log("GNN模型初始化完成")

    def forward(self, data):
        try:
            # 节点特征编码
            x = data.x  # [N, input_dim]
            edge_index = data.edge_index
            
            # 检查输入数据是否有NaN值
            if torch.isnan(x).any():
                log("警告: 输入特征包含NaN值，将被替换为0")
                x = torch.nan_to_num(x, nan=0.0)
            
            # 编码器处理
            x = self.encoder(x)  # [N, hidden]
            
            # 依次应用GNN层，每层之后进行非线性变换、批归一化、dropout和残差连接（如果可能）
            x_prev = x
            for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
                x_new = conv(x, edge_index)
                x_new = torch.relu(x_new)
                x_new = bn(x_new)
                x_new = self.dropout(x_new)
                
                # 添加残差连接（从第二层开始）
                if i > 0:
                    x_new = x_prev + x_new
                    
                x_prev = x_new
                x = x_new
            
            # 节点分类
            logits = self.cls_head(x)  # [N, 4]
            
            return logits
            
        except Exception as e:
            log(f"前向传播过程中出错: {str(e)}")
            # 安全地返回零张量作为后备方案
            return torch.zeros(data.x.size(0), len(Config.seq_vocab), device=data.x.device)

class RNADataset(Dataset):
    """RNA骨架坐标与序列数据集"""
    def __init__(self, coords_dir, seqs_dir, transform=None, max_samples=None):
        """
        初始化数据集
        
        Args:
            coords_dir: 骨架坐标文件目录
            seqs_dir: 序列文件目录
            transform: 预处理函数
            max_samples: 当设置时，限制加载的样本数量
        """
        self.coords_dir = coords_dir
        self.seqs_dir = seqs_dir
        self.transform = transform
        
        # 获取所有坐标文件
        self.coord_files = sorted(glob.glob(os.path.join(coords_dir, "*.npy")))
        
        # 如果指定了最大样本数，限制数量
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
            coord = np.load(coord_file)
            
            # 读取序列
            with open(seq_file, "r") as f:
                lines = f.readlines()
                # 剥除FASTA文件的头部（以>开始的行）
                seq = "".join([line.strip() for line in lines if not line.startswith(">")])
            
            # 确保坐标和序列长度匹配
            if coord.shape[0] != len(seq):
                log(f"警告: {coord_file} 坐标长度({coord.shape[0]})与序列长度({len(seq)})不匹配!")
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
            
            return graph
        except Exception as e:
            log(f"处理样本 {idx} 时出错: {str(e)}")
            # 创建一个简单的替代图
            dummy_coord = np.zeros((10, 7, 3))
            dummy_seq = "A" * 10
            return RNAGraphBuilder.build_graph(dummy_coord, dummy_seq)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20, save_dir='./'):
    """训练模型并返回训练历史"""
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    log(f"开始训练，总共{num_epochs}个epochs，设备:{device}")
    
    # 最佳模型性能
    best_val_acc = 0.0
    best_epoch = 0
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    log(f"检查点将保存到: {checkpoint_dir}")
    
    # 记录训练配置
    config_info = {
        "batch_size": Config.batch_size,
        "learning_rate": Config.lr,
        "hidden_dim": Config.hidden_dim,
        "epochs": num_epochs,
        "device": str(device),  # 将device对象转换为字符串
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as f:
        json.dump(config_info, f, indent=4)
    
    # 训练开始时间
    start_time = time.time()
    
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            log(f"开始 Epoch {epoch+1}/{num_epochs}")
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch_idx, data in enumerate(progress_bar):
                try:
                    # 将数据移到指定设备
                    data = data.to(device)
                    
                    # 清除之前的梯度
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = model(data)
                    
                    # 计算损失
                    loss = criterion(outputs, data.y)
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    optimizer.step()
                    
                    # 统计
                    train_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    batch_correct = (preds == data.y).sum().item()
                    batch_total = data.y.size(0)
                    train_correct += batch_correct
                    train_total += batch_total
                    
                    # 更新进度条
                    batch_acc = batch_correct / batch_total if batch_total > 0 else 0
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"})
                    
                    # 每50个批次记录一次日志
                    if (batch_idx + 1) % 50 == 0:
                        log(f"Epoch {epoch+1} - 批次 {batch_idx+1}/{len(train_loader)}: loss={loss.item():.4f}, acc={batch_acc:.4f}")
                except Exception as e:
                    log(f"训练批次 {batch_idx+1} 出错: {str(e)}")
                    continue
            
            # 计算平均损失和准确率
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for batch_idx, data in enumerate(progress_bar):
                    try:
                        data = data.to(device)
                        outputs = model(data)
                        loss = criterion(outputs, data.y)
                        
                        val_loss += loss.item()
                        preds = outputs.argmax(dim=1)
                        batch_correct = (preds == data.y).sum().item()
                        batch_total = data.y.size(0)
                        val_correct += batch_correct
                        val_total += batch_total
                        
                        # 更新进度条
                        batch_acc = batch_correct / batch_total if batch_total > 0 else 0
                        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"})
                    except Exception as e:
                        log(f"验证批次 {batch_idx+1} 出错: {str(e)}")
                        continue
            
            # 计算平均损失和准确率
            val_loss /= len(val_loader) if len(val_loader) > 0 else 1
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            # 更新学习率
            if scheduler:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                
                # 只有当学习率发生变化时才记录
                if new_lr != old_lr:
                    log(f"学习率从 {old_lr:.6f} 下降到 {new_lr:.6f}")
                else:
                    log(f"当前学习率: {new_lr:.6f}")
            
            # 保存当前epoch的性能
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # 计算每个epoch的训练时间
            epoch_time = time.time() - epoch_start_time
            
            # 打印当前epoch的性能
            log(f"Epoch {epoch+1}/{num_epochs} 完成 - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
                f"time: {epoch_time:.2f}s")
            
            # 每个epoch都保存检查点
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'history': history
            }, checkpoint_path)
            log(f"保存Epoch {epoch+1}的检查点")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_gnn_model.pth'))
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                log(f"保存最佳模型(Epoch {best_epoch})，验证集准确率: {val_acc:.4f}")
            
            # 保存并更新训练历史图
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:  # 每5个epoch或最后一个epoch保存一次
                plot_training_history(history, os.path.join(checkpoint_dir, f"training_history_e{epoch+1}.png"))
    
    except Exception as e:
        log(f"训练过程中断: {str(e)}")
        import traceback
        log(traceback.format_exc())
        # 保存中断时的模型
        interrupted_model_path = os.path.join(checkpoint_dir, 'interrupted_model.pth')
        torch.save(model.state_dict(), interrupted_model_path)
        log(f"已保存中断时的模型到 {interrupted_model_path}")
    
    # 训练总时间
    total_time = time.time() - start_time
    log(f"训练完成，总时间: {total_time/60:.2f}分钟")
    log(f"最佳性能: Epoch {best_epoch}，验证集准确率: {best_val_acc:.4f}")
    
    # 最终保存训练历史图
    plot_path = os.path.join(save_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    log(f"训练历史图已保存到 {plot_path}")
    
    return history, best_val_acc


def plot_training_history(history, save_path):
    """绘制并保存训练历史图表"""
    plt.figure(figsize=(12, 5))
    
    # 损失图
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 准确率图
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log(f"保存训练历史图到: {save_path}")

def calculate_recovery_rate(model, data_loader, device):
    """计算模型在数据加载器上的序列恢复率"""
    model.eval()
    total_correct = 0
    total_bases = 0
    
    # 序列类型准确率统计
    base_correct = {base: 0 for base in Config.seq_vocab}
    base_total = {base: 0 for base in Config.seq_vocab}
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="计算序列恢复率")
        for batch_idx, data in enumerate(progress_bar):
            try:
                data = data.to(device)
                outputs = model(data)
                preds = outputs.argmax(dim=1)
                
                # 总体准确率
                correct = (preds == data.y).sum().item()
                total_correct += correct
                total_bases += data.y.size(0)
                
                # 每种碱基的准确率
                for i, (pred, true) in enumerate(zip(preds, data.y)):
                    true_base = Config.idx_to_base[true.item()]
                    base_total[true_base] = base_total.get(true_base, 0) + 1
                    if pred == true:
                        base_correct[true_base] = base_correct.get(true_base, 0) + 1
                
                # 更新进度条
                current_rate = correct / data.y.size(0) if data.y.size(0) > 0 else 0
                overall_rate = total_correct / total_bases if total_bases > 0 else 0
                progress_bar.set_postfix({"batch_rate": f"{current_rate:.4f}", "overall": f"{overall_rate:.4f}"})
                
            except Exception as e:
                log(f"计算批次 {batch_idx+1} 恢复率时出错: {str(e)}")
                continue
    
    recovery_rate = total_correct / total_bases if total_bases > 0 else 0
    
    # 打印各碱基类型的恢复率
    log("各碱基类型的恢复率:")
    base_recovery = {}
    for base in Config.seq_vocab:
        rate = base_correct[base] / base_total[base] if base_total[base] > 0 else 0
        base_recovery[base] = rate
        log(f"  {base}: {rate:.4f} ({base_correct[base]}/{base_total[base]})")
    
    # 保存碱基恢复率到文件
    recovery_stats = {
        "overall": float(recovery_rate),
        "by_base": {base: float(rate) for base, rate in base_recovery.items()},
        "counts": {base: {"correct": base_correct[base], "total": base_total[base]} for base in Config.seq_vocab}
    }
    
    with open(os.path.join(LOG_DIR, "recovery_stats.json"), "w") as f:
        json.dump(recovery_stats, f, indent=4)
    
    return recovery_rate

def main():
    try:
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
        
        # 创建数据集
        log("加载数据集...")
        dataset = RNADataset(coords_dir, seqs_dir, transform=None)
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        log(f"训练集大小: {len(train_dataset)}")
        log(f"验证集大小: {len(val_dataset)}")
        
        # 创建数据加载器 - 减少worker数量，避免内存问题
        batch_size = Config.batch_size
        num_workers = 2  # 减少worker数量
        log(f"创建DataLoader，批量大小: {batch_size}，工作线程数: {num_workers}")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # 仅在GPU可用时使用锁页内存
            persistent_workers=True if num_workers > 0 else False  # 保持worker进程
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if num_workers > 0 else False
        )
        
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
        
        # 学习率调度器 - 适应不同版本的PyTorch
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,  # 每次将学习率减半
            patience=3,   # 3个epoch没有改善就降低学习率
            min_lr=1e-6   # 最小学习率
        )
        log(f"创建学习率调度器: 起始LR={Config.lr}, 最小LR=1e-6, 耐心值=3")
        
        # 训练模型
        log("开始训练模型...")
        history, best_val_acc = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            scheduler, 
            device, 
            num_epochs=Config.epochs,
            save_dir='.'
        )
        
        log(f"训练完成！最佳验证集准确率: {best_val_acc:.4f}")
        
        # 加载最佳模型并在验证集上评估
        log("加载最佳模型进行评估...")
        best_model_path = './best_gnn_model.pth'
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        # 计算序列恢复率
        log("计算序列恢复率...")
        recovery_rate = calculate_recovery_rate(model, val_loader, device)
        log(f"序列恢复率: {recovery_rate:.4f}")
        
        # 保存评估结果
        evaluation_results = {
            "best_val_acc": float(best_val_acc),
            "recovery_rate": float(recovery_rate),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open("evaluation_results.json", "w") as f:
            json.dump(evaluation_results, f, indent=4)
        
        log("评估结果已保存到 evaluation_results.json")
        
    except Exception as e:
        log(f"主函数执行出错: {str(e)}")
        import traceback
        log(traceback.format_exc())
    
    # 已通过log记录训练完成信息，这里不需要额外打印

if __name__ == "__main__":
    log("==== RNA序列生成模型训练开始 ====")
    main()
    log("==== 训练完成 ====")
