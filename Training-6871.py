import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import warnings
import re
import random
import math
from collections import defaultdict

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===== 高性能配置 =====
def get_enhanced_config():
    """结合两版本优势的高性能配置"""
    return {
        'backbone': 'resnet18',
        'num_modalities': 5,
        'num_tasks': 6,
        'feature_dim': 512,
        'batch_size': 24,              # 适中的batch size
        'learning_rate': 3e-5,         # 更保守的学习率
        'max_epochs': 100,             # 更多训练轮数
        'dataset_split': 0.8,          # 80%训练，20%测试
        'patience': 25,                # 更长的patience
        'tolerance': 0.5,
        'weight_decay': 3e-5,          # 适中的权重衰减
        'gradient_clip': 1.0,          # 适中的梯度裁剪
        'transformer_layers': 4,       # 更深的Transformer
        'num_heads': 8,               # 保持8个头
        'warmup_epochs': 15,          # 更长的warmup
        'label_smoothing': 0.05,      # 轻微标签平滑
        'mixup_alpha': 0.2,           # Mixup增强
        'cutmix_alpha': 1.0,          # CutMix增强
        'ema_decay': 0.999,           # 指数移动平均
        'focal_alpha': 0.25,          # Focal loss参数
        'focal_gamma': 2.0,
    }

# ===== 高级数据增强（结合Mixup/CutMix） =====
class EnhancedAugmentation:
    def __init__(self, is_training=True, config=None):
        self.is_training = is_training
        self.config = config or get_enhanced_config()
        
        if is_training:
            # 渐进式数据增强
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224), padding=16),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),  # 新增
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 新增
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # 新增
            ])
        else:
            # 测试时增强（TTA）
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, image):
        return self.transform(image)
    
    def mixup_data(self, x, y, alpha=0.2):
        """Mixup数据增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

# ===== 智能数据集类 =====
class EnhancedVISIADataset(Dataset):
    def __init__(self, data_dir, labels_file, is_training=True, num_modalities=5, config=None):
        self.data_dir = data_dir
        self.is_training = is_training
        self.num_modalities = num_modalities
        self.config = config or get_enhanced_config()
        
        # 加载标签
        self.labels_df = pd.read_csv(labels_file)
        first_column = self.labels_df.columns[0]
        self.subjects = self.labels_df[first_column].tolist()
        
        # 智能模态选择
        self.modalities = ['standard', 'uv', 'brown', 'red', 'uv_damage'][:num_modalities]
        
        # 高级标签预处理
        self._preprocess_labels_smart()
        self.task_columns = self.labels_df.columns[1:7].tolist()
        self.augmentation = EnhancedAugmentation(is_training=is_training, config=config)
        
        # 数据质量分析
        self._analyze_data_quality()
        
        print(f"Enhanced dataset: {len(self.subjects)} samples, {len(self.task_columns)} tasks")
        self._print_enhanced_stats()

    def _preprocess_labels_smart(self):
        print("Smart label preprocessing...")
        
        # 统计各列的分布
        label_stats = {}
        
        for col in self.labels_df.columns[1:7]:
            if col in self.labels_df.columns:
                # Q格式映射
                if self.labels_df[col].dtype == 'object':
                    mapping = {'Q1': 0.0, 'Q2': 1.0, 'Q3': 2.0, 'Q4': 3.0}
                    self.labels_df[col] = self.labels_df[col].map(mapping)
                
                # 数值化
                self.labels_df[col] = pd.to_numeric(self.labels_df[col], errors='coerce')
                
                # 记录统计信息
                original_mean = self.labels_df[col].mean()
                original_std = self.labels_df[col].std()
                label_stats[col] = {'mean': original_mean, 'std': original_std}
                
                # 智能填充：根据其他任务的相关性
                if self.labels_df[col].isna().any():
                    # 使用KNN填充
                    other_cols = [c for c in self.labels_df.columns[1:7] if c != col and c in self.labels_df.columns]
                    if other_cols:
                        imputer = KNNImputer(n_neighbors=3)
                        data_to_impute = self.labels_df[[col] + other_cols].values
                        imputed_data = imputer.fit_transform(data_to_impute)
                        self.labels_df[col] = imputed_data[:, 0]
                    else:
                        # 回退到中位数填充
                        self.labels_df[col] = self.labels_df[col].fillna(self.labels_df[col].median())
                
                # 确保范围[0,3]
                self.labels_df[col] = self.labels_df[col].clip(0.0, 3.0)
        
        self.label_stats = label_stats
        print("Smart label preprocessing complete")

    def _analyze_data_quality(self):
        self.data_quality = {}
        for i, subject in enumerate(self.subjects[:5]):  # 只检查前5个样本
            subject_num = self._extract_subject_number(subject)
            if isinstance(subject_num, int):
                subject_dir = os.path.join(self.data_dir, f"subject_{subject_num:03d}")
            else:
                subject_dir = os.path.join(self.data_dir, str(subject))
            
            if not os.path.exists(subject_dir):
                subject_dir = self._find_matching_folder(subject, subject_num)
            
            # 检查模态完整性
            available_modalities = []
            for modality in self.modalities:
                if self._check_modality_exists(subject_dir, modality):
                    available_modalities.append(modality)
            
            self.data_quality[subject] = {
                'available_modalities': len(available_modalities),
                'missing_modalities': len(self.modalities) - len(available_modalities)
            }
    
    def _check_modality_exists(self, subject_dir, modality):
        """检查模态文件是否存在"""
        if not os.path.exists(subject_dir):
            return False
        
        possible_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']
        for ext in possible_extensions:
            img_path = os.path.join(subject_dir, f"{modality}{ext}")
            if os.path.exists(img_path):
                return True
        return False
    
    def _print_enhanced_stats(self):
        print("Enhanced statistics:")

        for col in self.task_columns:
            values = self.labels_df[col].values
            skewness = pd.Series(values).skew()
            print(f"   {col}: mean{values.mean():.2f}, std{values.std():.2f}, "
                  f"skew{skewness:.2f}, range[{values.min():.1f}, {values.max():.1f}]")

        if self.data_quality:
            avg_modalities = np.mean([info['available_modalities'] for info in self.data_quality.values()])
            print(f"Average available modalities: {avg_modalities:.1f}/{self.num_modalities}")

    def _extract_subject_number(self, subject_name):
        if isinstance(subject_name, str):
            subject_match = re.search(r'subject[_\s]*(\d+)', subject_name, re.IGNORECASE)
            if subject_match:
                return int(subject_match.group(1))
            all_numbers = re.findall(r'\d+', subject_name)
            if all_numbers:
                return int(all_numbers[0])
        return subject_name

    def _find_matching_folder(self, subject_name, subject_num):
        if not os.path.exists(self.data_dir):
            return os.path.join(self.data_dir, str(subject_name))

        all_folders = [d for d in os.listdir(self.data_dir)
                       if os.path.isdir(os.path.join(self.data_dir, d))]

        if str(subject_name) in all_folders:
            return os.path.join(self.data_dir, str(subject_name))

        if isinstance(subject_num, int):
            for folder in all_folders:
                folder_nums = re.findall(r'\d+', folder)
                if folder_nums and int(folder_nums[0]) == subject_num:
                    return os.path.join(self.data_dir, folder)

        return os.path.join(self.data_dir, str(subject_name))

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_name = self.subjects[idx]
        subject_num = self._extract_subject_number(subject_name)

        if isinstance(subject_num, int):
            subject_dir = os.path.join(self.data_dir, f"subject_{subject_num:03d}")
        else:
            subject_dir = os.path.join(self.data_dir, str(subject_name))

        if not os.path.exists(subject_dir):
            subject_dir = self._find_matching_folder(subject_name, subject_num)

        # 加载多模态图像
        images = []
        for modality in self.modalities:
            image = self._load_modality_image(subject_dir, modality)
            image_tensor = self.augmentation(image)
            images.append(image_tensor)

        images = torch.stack(images, dim=0)
        task_labels = self._get_task_labels(idx)

        return {
            'images': images,
            'task_labels': torch.tensor(task_labels, dtype=torch.float32),
            'subject_id': subject_name
        }

    def _load_modality_image(self, subject_dir, modality):
        possible_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']

        for ext in possible_extensions:
            img_path = os.path.join(subject_dir, f"{modality}{ext}")
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                    return image
                except Exception:
                    pass

        # 回退到标准图像
        for ext in possible_extensions:
            standard_path = os.path.join(subject_dir, f"standard{ext}")
            if os.path.exists(standard_path):
                try:
                    image = Image.open(standard_path).convert('RGB')
                    return image
                except Exception:
                    pass

        return Image.new('RGB', (224, 224), (128, 128, 128))

    def _get_task_labels(self, idx):
        if idx < len(self.labels_df):
            label_row = self.labels_df.iloc[idx]
            task_labels = []
            for task_col in self.task_columns:
                try:
                    value = float(label_row[task_col])
                    value = max(0.0, min(3.0, value))
                    task_labels.append(value)
                except (KeyError, ValueError, TypeError):
                    # 使用该任务的统计中位数
                    default_val = self.label_stats.get(task_col, {}).get('mean', 1.5)
                    task_labels.append(min(max(default_val, 0.0), 3.0))
            return task_labels
        else:
            return [1.5] * len(self.task_columns)

# ===== 超级Transformer（自注意力+交叉注意力） =====
class SuperCrossImageTransformer(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_layers=4, num_modalities=5):
        super(SuperCrossImageTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # 可学习的位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, num_modalities, feature_dim) * 0.02)
        
        # 模态特定的线性变换
        self.modality_projections = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_modalities)
        ])
        
        # 双重注意力机制：自注意力 + 交叉注意力
        # 自注意力层
        self_attention_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.self_attention = nn.TransformerEncoder(self_attention_layer, num_layers=num_layers)
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(feature_dim)

    def forward(self, modality_features):
        batch_size = modality_features.size(0)
        
        # 模态特定投影
        projected_features = []
        for i in range(self.num_modalities):
            projected = self.modality_projections[i](modality_features[:, i])
            projected_features.append(projected)
        modality_features = torch.stack(projected_features, dim=1)
        
        # 添加位置编码
        modality_features = modality_features + self.pos_encoding
        
        # 自注意力
        self_attended = self.self_attention(modality_features)
        
        # 交叉注意力（每个模态关注其他模态）
        cross_attended_features = []
        for i in range(self.num_modalities):
            query = self_attended[:, i:i+1]  # [B, 1, D]
            key_value = self_attended  # [B, 5, D]
            
            cross_out, _ = self.cross_attention(query, key_value, key_value)
            cross_attended_features.append(cross_out.squeeze(1))
        
        cross_attended = torch.stack(cross_attended_features, dim=1)
        
        # 融合自注意力和交叉注意力的结果
        concat_features = torch.cat([self_attended, cross_attended], dim=-1)
        fused_features = self.fusion_net(concat_features)
        
        # 最终归一化
        fused_features = self.final_norm(fused_features)
        
        # 自适应权重融合
        attention_weights = F.softmax(torch.mean(fused_features, dim=-1), dim=-1)
        final_feature = torch.sum(fused_features * attention_weights.unsqueeze(-1), dim=1)
        
        return final_feature

# ===== 超级Pyramid Pooling =====
class SuperPyramidPooling(nn.Module):
    def __init__(self, feature_dim=512, pool_sizes=[1, 2, 3, 6]):
        super(SuperPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes
        self.feature_dim = feature_dim
        
        # 多尺度卷积层（并行处理不同尺度）
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(feature_dim, feature_dim // 4, 1, bias=False),
                nn.BatchNorm2d(feature_dim // 4),
                nn.GELU()
            ) for pool_size in pool_sizes
        ])
        
        # 注意力权重学习
        total_pyramid_dim = (feature_dim // 4) * len(pool_sizes)
        self.attention_conv = nn.Sequential(
            nn.Conv2d(total_pyramid_dim, total_pyramid_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(total_pyramid_dim // 4, len(pool_sizes), 1),
            nn.Sigmoid()
        )
        
        # 最终融合
        total_dim = feature_dim + total_pyramid_dim
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(feature_dim, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim)
        )
        
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramid_features = []
        
        # 并行处理多尺度特征
        for conv_layer in self.multi_scale_convs:
            pooled = conv_layer(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)
        
        # 拼接金字塔特征
        pyramid_concat = torch.cat(pyramid_features, dim=1)
        
        # 学习注意力权重
        attention_weights = self.attention_conv(pyramid_concat)
        
        # 应用注意力权重
        weighted_pyramid = []
        for i, feat in enumerate(pyramid_features):
            weight = attention_weights[:, i:i+1]
            weighted_feat = feat * weight
            weighted_pyramid.append(weighted_feat)
        
        weighted_pyramid_concat = torch.cat(weighted_pyramid, dim=1)
        
        # 与原始特征融合
        fused = torch.cat([x, weighted_pyramid_concat], dim=1)
        
        # 最终融合卷积
        output = self.fusion_conv(fused)
        
        # 残差连接
        output = output + x
        
        return output

# ===== 超级TSAA机制 =====
class SuperTSAA(nn.Module):
    def __init__(self, feature_dim=512, num_tasks=6):
        super(SuperTSAA, self).__init__()
        self.num_tasks = num_tasks
        self.feature_dim = feature_dim
        
        # 多层次总分预测
        self.total_score_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.LayerNorm(feature_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 4, 1)
        )
        
        # 任务间关系建模
        self.task_relation_matrix = nn.Parameter(torch.eye(num_tasks) + 
                                               0.1 * torch.randn(num_tasks, num_tasks))
        
        # 高级任务门控网络
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1 + feature_dim, feature_dim),  # 总分 + 特征
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
        
        # 任务特定的特征转换器
        self.task_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(feature_dim, feature_dim)
            ) for _ in range(num_tasks)
        ])
        
        # 任务协作注意力
        self.task_attention = nn.MultiheadAttention(
            feature_dim, num_heads=8, dropout=0.1, batch_first=True
        )

    def forward(self, features):
        batch_size = features.size(0)
        
        # 预测总分
        total_score = self.total_score_head(features)
        
        # 任务特定特征变换
        task_features = []
        for i in range(self.num_tasks):
            # 基础特征变换
            transformed = self.task_transformers[i](features)
            
            # 总分引导的门控
            gate_input = torch.cat([total_score, features], dim=-1)
            gate_weights = self.task_gates[i](gate_input)
            
            # 应用门控
            gated_feature = transformed * gate_weights + transformed * 0.1  # 残差
            task_features.append(gated_feature)
        
        # 任务协作：每个任务关注其他任务
        task_stack = torch.stack(task_features, dim=1)  # [B, num_tasks, feature_dim]
        
        # 应用任务关系矩阵
        relation_weights = F.softmax(self.task_relation_matrix, dim=-1)
        
        collaborative_features = []
        for i in range(self.num_tasks):
            # 当前任务作为query
            query = task_stack[:, i:i+1]  # [B, 1, feature_dim]
            
            # 其他任务作为key和value，权重由关系矩阵决定
            weighted_tasks = []
            for j in range(self.num_tasks):
                weight = relation_weights[i, j]
                weighted_task = task_stack[:, j] * weight
                weighted_tasks.append(weighted_task)
            
            key_value = torch.stack(weighted_tasks, dim=1)  # [B, num_tasks, feature_dim]
            
            # 协作注意力
            collab_out, _ = self.task_attention(query, key_value, key_value)
            collaborative_features.append(collab_out.squeeze(1))
        
        return collaborative_features, total_score.squeeze(-1)

# ===== 超级MultiTaskSkinNet =====
class SuperMultiTaskSkinNet(nn.Module):
    def __init__(self, num_modalities=5, num_tasks=6, config=None):
        super(SuperMultiTaskSkinNet, self).__init__()
        self.num_modalities = num_modalities
        self.num_tasks = num_tasks
        self.config = config or get_enhanced_config()
        
        # 改进的模态骨干网络
        self.modality_backbones = nn.ModuleList()
        for i in range(num_modalities):
            backbone = models.resnet18(pretrained=True)
            # 添加注意力模块到backbone
            backbone.layer4 = nn.Sequential(
                backbone.layer4,
                nn.AdaptiveAvgPool2d((7, 7)),  # 确保输出大小
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.GELU()
            )
            
            self.modality_backbones.append(nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
            ))
        
        # 超级组件
        self.super_transformer = SuperCrossImageTransformer(
            512, self.config['num_heads'], self.config['transformer_layers'], num_modalities
        )
        self.super_pyramid = SuperPyramidPooling(512)
        self.super_tsaa = SuperTSAA(512, num_tasks)
        
        # 最终预测头（集成多个预测器）
        self.main_task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            ) for _ in range(num_tasks)
        ])
        
        # 辅助预测头（集成学习）
        self.auxiliary_task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(128, 1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, images):
        batch_size = images.size(0)
        
        # 模态特征提取
        modality_features = []
        for i in range(self.num_modalities):
            modal_img = images[:, i]
            feature_map = self.modality_backbones[i](modal_img)  # [B, 512, 7, 7]
            pooled_features = F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
            modality_features.append(pooled_features)
        
        modality_features = torch.stack(modality_features, dim=1)  # [B, 5, 512]
        
        # 超级跨模态融合
        fused_features = self.super_transformer(modality_features)  # [B, 512]
        
        # 重塑并应用超级金字塔池化
        fused_feature_map = fused_features.view(batch_size, 512, 1, 1)
        fused_feature_map = F.interpolate(fused_feature_map, size=(7, 7), mode='nearest')
        enhanced_features = self.super_pyramid(fused_feature_map)
        enhanced_features = F.adaptive_avg_pool2d(enhanced_features, (1, 1)).flatten(1)
        
        # 超级TSAA
        task_features, total_score = self.super_tsaa(enhanced_features)
        
        # 主预测
        main_predictions = []
        for i, task_head in enumerate(self.main_task_heads):
            main_pred = task_head(task_features[i])
            main_predictions.append(main_pred)
        main_predictions = torch.cat(main_predictions, dim=1)
        
        # 辅助预测（用于集成）
        aux_predictions = []
        for i, aux_head in enumerate(self.auxiliary_task_heads):
            aux_pred = aux_head(enhanced_features)  # 直接使用增强特征
            aux_predictions.append(aux_pred)
        aux_predictions = torch.cat(aux_predictions, dim=1)
        
        # 集成预测
        ensemble_predictions = 0.8 * main_predictions + 0.2 * aux_predictions
        
        return {
            'task_predictions': ensemble_predictions,
            'main_predictions': main_predictions,
            'aux_predictions': aux_predictions,
            'total_score': total_score,
            'fused_features': enhanced_features
        }

# ===== 超级损失函数 =====
class SuperLoss(nn.Module):
    def __init__(self, num_tasks=6, config=None):
        super(SuperLoss, self).__init__()
        self.num_tasks = num_tasks
        self.config = config or get_enhanced_config()
        
        # 自适应任务权重
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
        
        # 多种损失函数
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.huber = nn.HuberLoss(reduction='none', delta=0.5)

    def focal_loss(self, predictions, targets, alpha=0.25, gamma=2.0):
        """Focal loss用于处理困难样本"""
        diff = torch.abs(predictions - targets)
        # 将回归问题转换为分类概率
        prob = torch.exp(-diff)
        focal_weight = alpha * (1 - prob) ** gamma
        loss = focal_weight * diff
        return loss

    def forward(self, outputs, targets, total_target=None):
        device = targets.device
        # 修复：不能直接给Parameter重新赋值，应该创建新变量
        task_weights = F.softmax(self.task_weights, dim=0).to(device)
        
        predictions = outputs['task_predictions']
        main_predictions = outputs['main_predictions'] 
        aux_predictions = outputs['aux_predictions']
        total_score = outputs.get('total_score')
        
        # 1. 主损失：加权组合多种损失
        smooth_l1_loss = self.smooth_l1(predictions, targets)
        huber_loss = self.huber(predictions, targets)
        focal_loss = self.focal_loss(predictions, targets, 
                                   self.config['focal_alpha'], 
                                   self.config['focal_gamma'])
        
        # 组合损失
        combined_loss = 0.5 * smooth_l1_loss + 0.3 * huber_loss + 0.2 * focal_loss
        weighted_loss = combined_loss * task_weights.unsqueeze(0)
        main_loss = weighted_loss.mean()
        
        total_loss = main_loss
        
        # 2. 辅助损失（集成学习）
        aux_loss = self.smooth_l1(aux_predictions, targets).mean()
        consistency_loss = self.mse(main_predictions, aux_predictions).mean()
        total_loss += 0.3 * aux_loss + 0.1 * consistency_loss
        
        # 3. 总分损失
        if total_score is not None and total_target is not None:
            total_score_loss = self.smooth_l1(total_score, total_target).mean()
            total_loss += 0.4 * total_score_loss
            
            # 预测一致性
            predicted_total = torch.mean(predictions, dim=1)
            prediction_consistency = self.mse(total_score, predicted_total).mean()
            total_loss += 0.2 * prediction_consistency
        
        # 4. 正则化损失
        # 预测多样性（避免退化解）
        pred_std = torch.std(predictions, dim=1).mean()
        diversity_loss = torch.relu(0.3 - pred_std)
        total_loss += 0.05 * diversity_loss
        
        # 任务权重正则化
        weight_reg = torch.var(task_weights)
        total_loss += 0.01 * weight_reg
        
        return total_loss

# ===== EMA模型 =====
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ===== 高级学习率调度器 =====
class AdvancedScheduler:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config
        self.base_lr = config['learning_rate']
        self.warmup_epochs = config['warmup_epochs']
        self.max_epochs = config['max_epochs']
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0

    def step(self, val_loss=None):
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing with restarts
            T_cur = self.current_epoch - self.warmup_epochs
            T_max = self.max_epochs - self.warmup_epochs
            lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * T_cur / T_max))
        
        # Adaptive adjustment based on validation loss
        if val_loss is not None:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter > 10:
                    lr *= 0.8  # Reduce learning rate
                    self.patience_counter = 0
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr

# ===== 测试时增强（TTA） =====
def test_time_augmentation(model, images, n_augs=5):
    """测试时增强"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # 原始预测
        output = model(images)
        predictions.append(output['task_predictions'])
        
        # 增强预测
        for _ in range(n_augs):
            # 轻微的几何变换
            augmented = images.clone()
            if random.random() > 0.5:
                augmented = torch.flip(augmented, dims=[-1])  # 水平翻转
            
            output = model(augmented)
            predictions.append(output['task_predictions'])
    
    # 平均预测
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    return avg_prediction

# ===== 超级训练函数 =====
def train_super_model(model, train_loader, val_loader, config, device='cuda'):
    model = model.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 调度器
    scheduler = AdvancedScheduler(optimizer, config)
    
    # EMA
    ema = EMAModel(model, decay=config['ema_decay'])
    
    # 损失函数
    criterion = SuperLoss(num_tasks=config['num_tasks'], config=config)
    
    # 数据增强
    augmentation = EnhancedAugmentation(config=config)
    
    best_val_acc = 0.0
    patience_counter = 0

    print(f"Super MultiTaskSkinNet training started")
    print(f"Config: Transformer={config['transformer_layers']} layers, "
          f"heads={config['num_heads']}, batch_size={config['batch_size']}")

    for epoch in range(config['max_epochs']):
        model.train()
        train_loss = 0.0
        train_losses = defaultdict(float)

        print(f"\nEpoch {epoch + 1}/{config['max_epochs']}")
        
        for i, batch in enumerate(train_loader):
            images = batch['images'].to(device, non_blocking=True)
            task_labels = batch['task_labels'].to(device, non_blocking=True)
            
            # Mixup增强（概率性应用）
            if random.random() < 0.3 and config.get('mixup_alpha', 0) > 0:
                images, targets_a, targets_b, lam = augmentation.mixup_data(
                    images, task_labels, config['mixup_alpha']
                )
                
                optimizer.zero_grad()
                outputs = model(images)
                total_target_a = torch.mean(targets_a, dim=1)
                total_target_b = torch.mean(targets_b, dim=1)
                
                loss_a = criterion(outputs, targets_a, total_target_a)
                loss_b = criterion(outputs, targets_b, total_target_b)
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                optimizer.zero_grad()
                outputs = model(images)
                total_target = torch.mean(task_labels, dim=1)
                loss = criterion(outputs, task_labels, total_target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
            optimizer.step()
            
            # 更新EMA
            ema.update()
            
            train_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # 验证阶段（使用EMA模型）
        ema.apply_shadow()
        val_acc, val_loss, val_mae, val_score_mae = evaluate_super_model(
            model, val_loader, device, config['tolerance']
        )
        ema.restore()
        
        current_lr = scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        print(f"Val Acc: {val_acc:.4f} | MAE: {val_mae:.4f} | Score MAE: {val_score_mae:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            ema.apply_shadow()
            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.shadow,
                    'epoch': epoch,
                    'best_acc': best_val_acc,
                    'config': config
                }, 'super_multitask_skinnet.pth')
                print(f"New best EMA model saved! Accuracy: {val_acc:.4f}")
            except Exception as e:
                print(f"Failed to save model: {e}")
            ema.restore()
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print(f"\nSuper training complete!")
    print(f"Best accuracy: {best_val_acc:.4f}")
    return best_val_acc

# ===== 超级评估函数 =====
def evaluate_super_model(model, test_loader, device='cuda', tolerance=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_total_preds = []
    all_total_labels = []
    total_loss = 0.0
    criterion = SuperLoss()
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device, non_blocking=True)
            task_labels = batch['task_labels'].to(device, non_blocking=True)
            
            # 使用TTA
            if len(images) <= 32:  # 只对小批次使用TTA以节省时间
                predictions = test_time_augmentation(model, images, n_augs=3)
                # 需要重新构造outputs字典用于loss计算
                outputs = model(images)
                outputs['task_predictions'] = predictions
            else:
                outputs = model(images)
            
            total_target = torch.mean(task_labels, dim=1)
            loss = criterion(outputs, task_labels, total_target)
            
            total_loss += loss.item()
            
            # 应用范围约束
            predictions = torch.clamp(outputs['task_predictions'], 0.0, 3.0)
            total_score = torch.clamp(outputs['total_score'], 0.0, 3.0)
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(task_labels.cpu().numpy())
            all_total_preds.append(total_score.cpu().numpy())
            all_total_labels.append(total_target.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_total_preds = np.concatenate(all_total_preds, axis=0)
    all_total_labels = np.concatenate(all_total_labels, axis=0)
    
    # 计算指标
    tolerance_correct = np.abs(all_preds - all_labels) <= tolerance
    avg_accuracy = np.mean(tolerance_correct)
    mae = np.mean(np.abs(all_preds - all_labels))
    score_mae = np.mean(np.abs(all_total_preds - all_total_labels))
    
    task_accuracies = np.mean(tolerance_correct, axis=0)
    task_maes = np.mean(np.abs(all_preds - all_labels), axis=0)

    print(f"Task accuracies: {[f'{acc:.3f}' for acc in task_accuracies]}")
    print(f"Task MAEs: {[f'{mae:.3f}' for mae in task_maes]}")
    print(f"Prediction range: [{all_preds.min():.3f}, {all_preds.max():.3f}]")
    print(f"Label distribution: mean{all_labels.mean():.3f}, std{all_labels.std():.3f}")
    print(f"Prediction distribution: mean{all_preds.mean():.3f}, std{all_preds.std():.3f}")

    avg_loss = total_loss / len(test_loader)

    return avg_accuracy, avg_loss, mae, score_mae


def main():
    print("Super MultiTaskSkinNet - High-performance implementation")
    set_seed(42)

    config = get_enhanced_config()
    print(f"Super config: {config}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据路径
    data_dir = "New_output"
    labels_file = "Skin_lable.csv"
    
    # 创建超级数据集
    full_dataset = EnhancedVISIADataset(
        data_dir, 
        labels_file, 
        is_training=True, 
        num_modalities=config['num_modalities'],
        config=config
    )
    
    # 数据集划分
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)),
        test_size=1-config['dataset_split'],
        random_state=42,
        stratify=None  # 由于是回归任务，不进行分层
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    
    val_dataset_full = EnhancedVISIADataset(
        data_dir, 
        labels_file, 
        is_training=False, 
        num_modalities=config['num_modalities'],
        config=config
    )
    val_dataset = torch.utils.data.Subset(val_dataset_full, test_indices)

    print(f"Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}")
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 创建超级模型
    model = SuperMultiTaskSkinNet(
        num_modalities=config['num_modalities'],
        num_tasks=config['num_tasks'],
        config=config
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    best_acc = train_super_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    print(f"\nFinal evaluation:")

    model_path = 'super_multitask_skinnet.pth'
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Successfully loaded best model")
        except Exception as e:
            print(f"Loading failed, using current model: {e}")
    
    final_acc, final_loss, final_mae, final_score_mae = evaluate_super_model(
        model, val_loader, device, config['tolerance']
    )

    print(f"\nSuper training complete!")
    print(f"Best accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Final accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"Final MAE: {final_mae:.4f}")
    print(f"Score MAE: {final_score_mae:.4f}")

    prev_acc_1 = 0.634
    prev_acc_2 = 0.6439
    improvement_1 = final_acc - prev_acc_1
    improvement_2 = final_acc - prev_acc_2

    print(f"\nPerformance comparison:")
    print(f"   Improvement vs v1: {improvement_1:.4f} ({improvement_1*100:.2f}%)")
    print(f"   Improvement vs v2: {improvement_2:.4f} ({improvement_2*100:.2f}%)")

    paper_target = 0.6712
    if final_acc > paper_target:
        print("Exceeded paper target level!")
    elif final_acc > paper_target - 0.01:
        print("Very close to paper level!")
    elif final_acc > max(prev_acc_1, prev_acc_2) + 0.005:
        print("Significant improvement!")
    else:
        print("Continuous improvement...")

    return final_acc, final_mae, final_score_mae

if __name__ == "__main__":
    accuracy, mae, score_mae = main()