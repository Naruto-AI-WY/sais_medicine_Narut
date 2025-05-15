import numpy as np
import torch
import torch.nn as nn
import os
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import glob

class Config:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    lr = 0.01
    epochs = 20
    seq_vocab = "AUCG"
    coord_dims = 7  # 7个骨架点
    hidden_dim = 256  # 增加隐藏层维度
    k_neighbors = 8  # 增加近邻节点数以捕捉更多结构信息
    dropout = 0.2  # 添加dropout防止过拟合
    edge_features = True  # 使用边特征
    spatial_neighbors = True  # 使用空间近邻

class GNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 增强特征编码
        self.encoder = nn.Sequential(
            nn.Linear(7*3, Config.hidden_dim),
            nn.BatchNorm1d(Config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.hidden_dim, Config.hidden_dim),
            nn.ReLU()
        )
        
        # 增加GNN层数和复杂度
        from torch_geometric.nn import GATConv, GINConv, ResGatedGraphConv
        
        # GNN层组合使用不同类型的卷积
        self.conv1 = GCNConv(Config.hidden_dim, Config.hidden_dim)
        self.conv2 = GATConv(Config.hidden_dim, Config.hidden_dim, heads=4, dropout=Config.dropout)
        self.conv3 = ResGatedGraphConv(Config.hidden_dim*4, Config.hidden_dim)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(Config.hidden_dim)
        self.bn2 = nn.BatchNorm1d(Config.hidden_dim*4)
        self.bn3 = nn.BatchNorm1d(Config.hidden_dim)
        
        # 分类头，增加深度
        self.cls_head = nn.Sequential(
            nn.Linear(Config.hidden_dim, Config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.hidden_dim // 2, len(Config.seq_vocab))
        )
        
        # 全局特征
        self.global_features = nn.Parameter(torch.randn(1, Config.hidden_dim))

    def forward(self, data):
        # 节点特征编码
        x = self.encoder(data.x)  # [N, hidden]
        
        # 图卷积层 1 - 带残差连接
        identity = x
        x = self.conv1(x, data.edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = x + identity  # 残差连接
        x = F.dropout(x, p=Config.dropout, training=self.training)
        
        # 图卷积层 2 - GAT多头注意力
        identity = x
        x = self.conv2(x, data.edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        # 图卷积层 3 - 门控GNN
        x = self.conv3(x, data.edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 添加全局上下文信息
        global_context = self.global_features.expand(x.size(0), -1)
        x = x + global_context * 0.1  # 轻微融合全局信息
        
        # 节点分类
        logits = self.cls_head(x)  # [N, 4]
        return logits

class RNAGraphBuilder:
    @staticmethod
    def build_graph(coord, seq):
        """将坐标和序列转换为图结构，增强版"""
        num_nodes = coord.shape[0]

        # 1. 增强节点特征工程
        # 基本坐标特征
        basic_feat = coord.reshape(num_nodes, -1)  # [N, 7*3]
        
        # 计算二面角特征（如果序列长度足够）
        dihedral_feat = np.zeros((num_nodes, 6))  # 假设6个二面角特征
        if num_nodes > 3:
            # 选择C4'原子(假设是第4个原子)计算二面角
            c4_coords = coord[:, 3, :]
            for i in range(1, num_nodes-2):
                try:
                    # 计算连续4个C4'原子形成的二面角
                    p1, p2, p3, p4 = c4_coords[i-1:i+3]
                    
                    # 检查坐标是否有效
                    if np.any(np.isnan(p1)) or np.any(np.isnan(p2)) or np.any(np.isnan(p3)) or np.any(np.isnan(p4)):
                        continue
                        
                    v1, v2, v3 = p2-p1, p3-p2, p4-p3
                    
                    # 检查向量是否为零
                    if np.all(np.isclose(v1, 0)) or np.all(np.isclose(v2, 0)) or np.all(np.isclose(v3, 0)):
                        continue
                        
                    n1 = np.cross(v1, v2)
                    n2 = np.cross(v2, v3)
                    
                    # 检查法向量是否为零向量
                    n1_norm = np.linalg.norm(n1)
                    n2_norm = np.linalg.norm(n2)
                    if n1_norm < 1e-6 or n2_norm < 1e-6:
                        continue
                        
                    # 计算点积并归一化，使用clip确保在[-1, 1]范围内
                    cos_angle = np.clip(np.dot(n1, n2) / (n1_norm * n2_norm), -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    
                    # 存储二面角特征
                    dihedral_feat[i, 0] = np.sin(angle)
                    dihedral_feat[i, 1] = np.cos(angle)
                except Exception as e:
                    # 如果计算出错，简单跳过
                    continue
        
        # 计算节点间距离统计特征
        dist_feat = np.zeros((num_nodes, 4))  # 距离统计特征
        for i in range(num_nodes):
            # 使用C4'原子计算与相邻节点的距离
            c4_i = coord[i, 3, :]
            distances = []
            for j in range(max(0, i-10), min(num_nodes, i+11)):
                if j != i:
                    c4_j = coord[j, 3, :]
                    distances.append(np.linalg.norm(c4_i - c4_j))
            
            if distances:
                dist_feat[i, 0] = np.mean(distances)  # 平均距离
                dist_feat[i, 1] = np.std(distances)   # 距离标准差
                dist_feat[i, 2] = np.min(distances)   # 最小距离
                dist_feat[i, 3] = np.max(distances)   # 最大距离
        
        # 组合所有特征
        all_features = np.concatenate([basic_feat, dihedral_feat, dist_feat], axis=1)
        
        # 归一化特征
        mean = np.mean(all_features, axis=0, keepdims=True)
        std = np.std(all_features, axis=0, keepdims=True) + 1e-6
        norm_features = (all_features - mean) / std
        
        # 转换为张量
        x = torch.tensor(norm_features, dtype=torch.float32)

        # 2. 增强边构建
        edge_index = []
        edge_attr = []  # 边特征
        
        # 序列顺序连接（骨架连接）
        for i in range(num_nodes):
            # 连接前k和后k个节点（序列近邻）
            neighbors = list(range(max(0, i-Config.k_neighbors), i)) + \
                      list(range(i+1, min(num_nodes, i+1+Config.k_neighbors)))
            
            for j in neighbors:
                if 0 <= j < num_nodes:
                    # 添加双向边
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    # 计算边特征：相对位置编码
                    pos_enc = abs(i - j)
                    # 计算两个节点主原子间的欧氏距离
                    c4_i = coord[i, 3, :]
                    c4_j = coord[j, 3, :]
                    distance = np.linalg.norm(c4_i - c4_j)
                    
                    # 边特征：[相对位置编码, 距离, 是否是序列相邻]
                    edge_feat = [pos_enc, distance, 1.0 if abs(i-j)==1 else 0.0]
                    edge_attr.append(edge_feat)
                    edge_attr.append(edge_feat)  # 双向边共享特征
        
        # 空间近邻连接 - 如果启用
        if Config.spatial_neighbors:
            # 计算所有C4'原子之间的距离矩阵
            c4_coords = coord[:, 3, :]  # 假设索引3是C4'原子
            dist_matrix = np.zeros((num_nodes, num_nodes))
            
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    dist = np.linalg.norm(c4_coords[i] - c4_coords[j])
                    dist_matrix[i, j] = dist_matrix[j, i] = dist
            
            # 为每个节点添加最近的K个空间邻居（非序列邻居）
            spatial_k = min(5, num_nodes-1)  # 空间近邻数量
            for i in range(num_nodes):
                # 获取距离排序
                dist_i = dist_matrix[i]
                # 排除自身和已经作为序列邻居的节点
                seq_neighbors = set(range(max(0, i-Config.k_neighbors), i)) | \
                               set(range(i+1, min(num_nodes, i+1+Config.k_neighbors)))
                candidates = [(j, dist_i[j]) for j in range(num_nodes) 
                              if j != i and j not in seq_neighbors]
                
                # 按距离排序
                candidates.sort(key=lambda x: x[1])
                spatial_neighbors = candidates[:spatial_k]
                
                # 添加空间近邻边
                for j, dist in spatial_neighbors:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    # 边特征：[相对位置编码(较大值表示非序列邻居), 距离, 是否是序列相邻]
                    edge_feat = [100, dist, 0.0]  # 非序列相邻
                    edge_attr.append(edge_feat)
                    edge_attr.append(edge_feat)  # 双向边共享特征

        # 转换为张量
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # 边特征（如果启用）
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if Config.edge_features else None
        
        # 节点标签
        y = torch.tensor([Config.seq_vocab.index(c) for c in seq], dtype=torch.long)

        # 创建包含额外特征的图数据对象
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
        if edge_attr is not None:
            data.edge_attr = edge_attr
            
        return data

class RNASequenceGenerator:
    def __init__(self, model_path):
        # 初始化多个模型实例做集成
        self.model = GNNModel().to(Config.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=Config.device, weights_only=True)
        )
        self.model.eval()
        
        # RNA序列的生物物理约束
        self.valid_pairs = {
            'A': ['U'],  # A配对U
            'U': ['A', 'G'],  # U配对A或G
            'C': ['G'],  # C配对G
            'G': ['C', 'U']   # G配对C或U
        }
        
        # 初始化序列模式字典，用于通用的RNA motif识别
        self.known_motifs = {
            'GNRA': {'pattern': 'G[AUCG][AG]A', 'score': 0.8},  # GNRA发夹
            'UNCG': {'pattern': 'U[AUCG]CG', 'score': 0.7},     # UNCG发夹
            'CUUG': {'pattern': 'CUUG', 'score': 0.6}            # CUUG发夹
        }
        
        # 设置随机种子确保结果可重复
        np.random.seed(Config.seed)

    def generate_sequences(self, coord_data, num_seq=5, temperature=1.0, top_k=3):
        """
        生成候选RNA序列，增强版
        :param coord_data: numpy数组 [L, 7, 3]
        :param num_seq: 需要生成的序列数量
        :param temperature: 温度参数控制多样性
        :param top_k: 每个位置只考虑top_k高概率的碰基
        :return: 生成的序列列表
        """
        # 转换为图数据
        graph = self._preprocess_data(coord_data)
        graph = graph.to(Config.device)
        
        # 获取概率分布
        with torch.no_grad():
            logits = self.model(graph)
            base_probs = F.softmax(logits / temperature, dim=1)  # [L, 4]
        
        # 添加不同的采样策略生成候选序列
        candidates = []
        
        # 策略 1: 多温度采样，抓取不同的定义尺度
        temp_values = [0.8, 1.0, 1.2]
        for temp in temp_values:
            probs = F.softmax(logits / temp, dim=1)
            for _ in range(2):
                seq = self._advanced_sampling(probs, top_k=4)  # 使用增强的采样方法
                candidates.append(seq)
        
        # 策略 2: 上下文感知的自回归解码
        for _ in range(2):
            seq = self._autoregressive_sampling(base_probs, top_k=3)
            candidates.append(seq)
            
        # 策略 3: 使用最大概率解码（贪心解码）
        greedy_seq = self._greedy_decode(base_probs)
        candidates.append(greedy_seq)
        
        # 评估并排序候选序列
        scored_candidates = []
        for seq in candidates:
            score = self._evaluate_sequence(seq, coord_data)
            scored_candidates.append((seq, score))
            
        # 按评分排序
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 返回评分最高的num_seq个方案
        result = []
        for seq, _ in scored_candidates[:num_seq]:
            if seq not in result:
                result.append(seq)
                
        # 确保结果数量足够
        while len(result) < num_seq and len(candidates) > 0:
            extra_seq = self._advanced_sampling(base_probs, top_k=4)
            if extra_seq not in result:
                result.append(extra_seq)
                
        return result

    def _preprocess_data(self, coord):
        """增强版预处理坐标数据为图结构"""
        # 创建伪序列
        dummy_seq = "A" * coord.shape[0]
        
        # 完成数据清洗
        # 检查坐标数据中的异常值并替换
        clean_coord = np.copy(coord)
        clean_coord = np.nan_to_num(clean_coord, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 声明数据这里有错误值处理
        mask = np.isclose(np.sum(np.abs(clean_coord), axis=(1, 2)), 0)
        if np.any(mask):
            # 对于全零坐标，使用前后点的平均值进行插值
            for i in range(len(mask)):
                if mask[i]:
                    # 找到前后有效的坐标进行插值
                    valid_indices = np.where(~mask)[0]
                    if len(valid_indices) > 0:
                        # 找最近的有效点
                        dists = np.abs(valid_indices - i)
                        nearest_idx = valid_indices[np.argmin(dists)]
                        clean_coord[i] = clean_coord[nearest_idx]
        
        return RNAGraphBuilder.build_graph(clean_coord, dummy_seq)
    
    def _evaluate_sequence(self, seq, coord_data):
        """评估序列质量的函数"""
        score = 0.0
        
        # 1. 检查碰基配对是否合理
        pair_score = 0
        seq_len = len(seq)
        
        for i in range(seq_len):
            for j in range(i+4, seq_len):  # 至少3个碰基的间隔才能配对
                # 检查是否可以形成配对
                if seq[j] in self.valid_pairs.get(seq[i], []):
                    # 计算带关系的原子间的距离
                    phosphate_i = coord_data[i, 0, :]  # 假设第一个原子是碰酶
                    phosphate_j = coord_data[j, 0, :]
                    # 如果距离适合配对，增加分数
                    dist = np.linalg.norm(phosphate_i - phosphate_j)
                    if 10.0 < dist < 20.0:  # RNA碰基对典型距离范围
                        pair_score += 1
        
        # 归一化配对得分
        max_possible_pairs = seq_len // 2
        pair_score = pair_score / max(1, max_possible_pairs)
        score += pair_score * 0.4  # 权重0.4
        
        # 2. 检查序列中的RNA motif
        motif_score = 0
        import re
        for name, motif in self.known_motifs.items():
            if re.search(motif['pattern'], seq):
                motif_score += motif['score']
        motif_score = min(1.0, motif_score)  # 限制最高分
        score += motif_score * 0.3  # 权重0.3
        
        # 3. GC含量应在合理范围
        gc_count = seq.count('G') + seq.count('C')
        gc_content = gc_count / len(seq)
        # RNA序列的GC含量一般在0.3-0.7之间较合理
        if 0.3 <= gc_content <= 0.7:
            gc_score = 1.0
        else:
            gc_score = 1.0 - 2 * min(abs(gc_content - 0.3), abs(gc_content - 0.7))
        score += gc_score * 0.2  # 权重0.2
        
        # 4. 序列的均衡性评估
        char_counts = {c: seq.count(c)/len(seq) for c in "AUCG"}
        entropy = -sum(p * np.log2(p+1e-10) for p in char_counts.values()) / 2.0  # 最大值约2
        score += entropy * 0.1  # 权重0.1
        
        return score
            
    def _advanced_sampling(self, probs, top_k=3):
        """增强版采样方法，考虑前后上下文信息"""
        seq_len = probs.size(0)
        seq = []
        
        for i in range(seq_len):
            node_probs = probs[i].clone()
            
            # 如果不是第一个位置，循环尝试考虑上下文信息
            if i > 0:
                prev_base = seq[-1]  # 前一个碰基
                
                # 细节1: 根据已知的RNA相邻碰基偏好调整当前概率
                if prev_base == 'A':
                    # A之后不太可能AA，更可能是AU/AG
                    node_probs[Config.seq_vocab.index('A')] *= 0.7
                    node_probs[Config.seq_vocab.index('U')] *= 1.2
                    
                elif prev_base == 'G':
                    # G后面序列偏好C/U
                    node_probs[Config.seq_vocab.index('C')] *= 1.2
                    node_probs[Config.seq_vocab.index('U')] *= 1.1
                    
                # 防止3个以上相同碰基
                if i >= 2 and seq[-1] == seq[-2]:
                    repeat_base = seq[-1]
                    # 压缩重复碰基的概率
                    node_probs[Config.seq_vocab.index(repeat_base)] *= 0.5
            
            # 应用top-k筛选并重新归一化
            topk_probs, topk_indices = torch.topk(node_probs, min(top_k, len(Config.seq_vocab)))
            norm_probs = topk_probs / topk_probs.sum()
            
            # 采样
            chosen = np.random.choice(topk_indices.cpu().numpy(), p=norm_probs.cpu().numpy())
            seq.append(Config.seq_vocab[chosen])
        
        return "".join(seq)
    
    def _autoregressive_sampling(self, probs, top_k=3):
        """自回归采样，递进地生成序列"""
        seq_len = probs.size(0)
        seq = []
        
        # 使用段首、段尾的特殊处理
        # RNA序列往往在开头和结尾有一些特殊的模式
        
        # 先选定第一个碚基(由于常见RNA序列开头偏好G)
        first_probs = probs[0].clone()
        first_probs[Config.seq_vocab.index('G')] *= 1.3  # 提升G在开头的概率
        topk_probs, topk_indices = torch.topk(first_probs, min(top_k, len(Config.seq_vocab)))
        norm_probs = topk_probs / topk_probs.sum()
        chosen = np.random.choice(topk_indices.cpu().numpy(), p=norm_probs.cpu().numpy())
        seq.append(Config.seq_vocab[chosen])
        
        # 生成中间部分
        for i in range(1, seq_len-1):
            node_probs = probs[i].clone()
            
            # 基于已生成序列的上下文调整概率
            prev_bases = "".join(seq[-min(3, len(seq)):])
            
            # 特殊模式的处理
            if prev_bases == "GA" or prev_bases == "GG":
                # GNRA motif的第三个碚基偏好A或G
                node_probs[Config.seq_vocab.index('A')] *= 1.2
                node_probs[Config.seq_vocab.index('G')] *= 1.2
            
            # 应用top-k筛选
            topk_probs, topk_indices = torch.topk(node_probs, min(top_k, len(Config.seq_vocab)))
            norm_probs = topk_probs / topk_probs.sum()
            chosen = np.random.choice(topk_indices.cpu().numpy(), p=norm_probs.cpu().numpy())
            seq.append(Config.seq_vocab[chosen])
        
        # 最后一个序列位置
        if seq_len > 1:
            last_probs = probs[seq_len-1].clone()
            # 如果是GNN模式的结尾
            if "".join(seq[-3:]) == "GNR":  # 假设我们已经生成了GNR
                last_probs[Config.seq_vocab.index('A')] *= 3.0  # 大幅提升A的概率
            
            topk_probs, topk_indices = torch.topk(last_probs, min(top_k, len(Config.seq_vocab)))
            norm_probs = topk_probs / topk_probs.sum()
            chosen = np.random.choice(topk_indices.cpu().numpy(), p=norm_probs.cpu().numpy())
            seq.append(Config.seq_vocab[chosen])
        
        return "".join(seq)
    
    def _greedy_decode(self, probs):
        """贪心解码，始终选取概率最高的碰基"""
        seq = []
        for node_probs in probs:
            chosen = torch.argmax(node_probs).item()
            seq.append(Config.seq_vocab[chosen])
        return "".join(seq)

# 使用示例
if __name__ == "__main__":
    print(f"\n[INFO] 启动RNA序列预测器 (运行设备: {Config.device})")
    print(f"[INFO] 模型配置: hidden_dim={Config.hidden_dim}, k_neighbors={Config.k_neighbors}")
    
    # 设置随机种子确保可重现性
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.seed)
    
    try:
        # 加载生成器
        print("[INFO] 加载模型...")
        generator = RNASequenceGenerator("best_gnn_model.pth")
        
        # 初始化结果存储
        result = {
            "pdb_id": [],
            "seq": []
        }
        
        # 获取所有输入文件
        npy_files = glob.glob("/saisdata/coords/*.npy")
        total_files = len(npy_files)
        print(f"[INFO] 发现{total_files}个坐标文件需要处理")
        
        # 处理每个文件
        for idx, npy in enumerate(npy_files):
            try:
                id_name = os.path.basename(npy).split(".")[0]
                print(f"[INFO] 处理文件 {idx+1}/{total_files}: {id_name}")
                
                # 加载并预处理坐标数据
                coord = np.load(npy)  # [L, 7, 3]
                
                # 检查数据有效性
                if coord.size == 0 or np.all(np.isnan(coord)):
                    print(f"[WARNING] 文件 {id_name} 坐标全为NaN，使用预设序列")
                    # 对于无效数据，生成一个均匀的随机序列
                    seq_len = 10  # 默认长度
                    result["pdb_id"].append(id_name)
                    result["seq"].append("".join(np.random.choice(list(Config.seq_vocab), size=seq_len)))
                    continue
                
                # 清理数据
                coord = np.nan_to_num(coord, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 生成多个候选序列并选择最佳的那个
                print(f"[INFO] 预测序列（序列长度: {coord.shape[0]}bp）")
                
                # 强化版生成器会生成多个候选项并返回最佳的那个
                candidates = generator.generate_sequences(
                    coord,
                    num_seq=1,  # 需要返回的最终序列数量
                    temperature=0.8,  # 适度多样性
                    top_k=4          # 每个位置考虑前4个可能
                )
                
                # 添加结果
                result["pdb_id"].append(id_name)
                result["seq"].append(candidates[0])
                
                # 这里可以打印一些序列统计信息
                seq = candidates[0]
                gc_content = (seq.count('G') + seq.count('C')) / len(seq) if seq else 0
                print(f"[INFO] 生成序列: {seq[:20]}... (GC含量: {gc_content:.2f})")
                
            except Exception as e:
                print(f"[ERROR] 处理文件 {id_name} 时出错: {str(e)}")
                # 对于错误情况，生成一个安全的默认序列
                result["pdb_id"].append(id_name)
                default_seq = "GAAAUUUCGC"  # 一个平衡的默认序列
                result["seq"].append(default_seq)
        
        # 转换为数据帧并保存
        result_df = pd.DataFrame(result)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname("/saisresult/submit.csv"), exist_ok=True)
        
        # 保存结果
        result_df.to_csv("/saisresult/submit.csv", index=False)
        print(f"\n[SUCCESS] 处理完成! 结果已保存至 /saisresult/submit.csv")
        print(f"[INFO] 共生成 {len(result_df)} 条序列预测")
        
    except Exception as e:
        print(f"[CRITICAL ERROR] 程序出现全局错误: {str(e)}")
        # 对于全局错误，我们仍然尝试生成一个空白的结果文件
        empty_result = pd.DataFrame({"pdb_id": [], "seq": []})
        try:
            os.makedirs(os.path.dirname("/saisresult/submit.csv"), exist_ok=True)
            empty_result.to_csv("/saisresult/submit.csv", index=False)
            print("[INFO] 已生成空结果文件")
        except:
            print("[CRITICAL] 无法生成结果文件!")



