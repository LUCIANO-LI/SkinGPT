import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import os
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')


class SuperCrossImageTransformer(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_layers=4, num_modalities=5):
        super(SuperCrossImageTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        self.pos_encoding = nn.Parameter(torch.randn(1, num_modalities, feature_dim) * 0.02)
        
        self.modality_projections = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_modalities)
        ])
        
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
        
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.final_norm = nn.LayerNorm(feature_dim)

    def forward(self, modality_features):
        batch_size = modality_features.size(0)
        
        projected_features = []
        for i in range(self.num_modalities):
            projected = self.modality_projections[i](modality_features[:, i])
            projected_features.append(projected)
        modality_features = torch.stack(projected_features, dim=1)
        
        modality_features = modality_features + self.pos_encoding
        
        self_attended = self.self_attention(modality_features)
        
        cross_attended_features = []
        for i in range(self.num_modalities):
            query = self_attended[:, i:i+1]
            key_value = self_attended
            
            cross_out, _ = self.cross_attention(query, key_value, key_value)
            cross_attended_features.append(cross_out.squeeze(1))
        
        cross_attended = torch.stack(cross_attended_features, dim=1)
        
        concat_features = torch.cat([self_attended, cross_attended], dim=-1)
        fused_features = self.fusion_net(concat_features)
        
        fused_features = self.final_norm(fused_features)
        
        attention_weights = F.softmax(torch.mean(fused_features, dim=-1), dim=-1)
        final_feature = torch.sum(fused_features * attention_weights.unsqueeze(-1), dim=1)
        
        return final_feature

class SuperPyramidPooling(nn.Module):
    def __init__(self, feature_dim=512, pool_sizes=[1, 2, 3, 6]):
        super(SuperPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes
        self.feature_dim = feature_dim
        
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(feature_dim, feature_dim // 4, 1, bias=False),
                nn.BatchNorm2d(feature_dim // 4),
                nn.GELU()
            ) for pool_size in pool_sizes
        ])
        
        total_pyramid_dim = (feature_dim // 4) * len(pool_sizes)
        self.attention_conv = nn.Sequential(
            nn.Conv2d(total_pyramid_dim, total_pyramid_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(total_pyramid_dim // 4, len(pool_sizes), 1),
            nn.Sigmoid()
        )
        
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
        
        for conv_layer in self.multi_scale_convs:
            pooled = conv_layer(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)
        
        pyramid_concat = torch.cat(pyramid_features, dim=1)
        
        attention_weights = self.attention_conv(pyramid_concat)
        
        weighted_pyramid = []
        for i, feat in enumerate(pyramid_features):
            weight = attention_weights[:, i:i+1]
            weighted_feat = feat * weight
            weighted_pyramid.append(weighted_feat)
        
        weighted_pyramid_concat = torch.cat(weighted_pyramid, dim=1)
        
        fused = torch.cat([x, weighted_pyramid_concat], dim=1)
        
        output = self.fusion_conv(fused)
        
        output = output + x
        
        return output

class SuperTSAA(nn.Module):
    def __init__(self, feature_dim=512, num_tasks=6):
        super(SuperTSAA, self).__init__()
        self.num_tasks = num_tasks
        self.feature_dim = feature_dim
        
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
        
        self.task_relation_matrix = nn.Parameter(torch.eye(num_tasks) + 
                                               0.1 * torch.randn(num_tasks, num_tasks))
        
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1 + feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
        
        self.task_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(feature_dim, feature_dim)
            ) for _ in range(num_tasks)
        ])
        
        self.task_attention = nn.MultiheadAttention(
            feature_dim, num_heads=8, dropout=0.1, batch_first=True
        )

    def forward(self, features):
        batch_size = features.size(0)
        
        total_score = self.total_score_head(features)
        
        task_features = []
        for i in range(self.num_tasks):
            transformed = self.task_transformers[i](features)
            
            gate_input = torch.cat([total_score, features], dim=-1)
            gate_weights = self.task_gates[i](gate_input)
            
            gated_feature = transformed * gate_weights + transformed * 0.1
            task_features.append(gated_feature)
        
        task_stack = torch.stack(task_features, dim=1)
        
        relation_weights = F.softmax(self.task_relation_matrix, dim=-1)
        
        collaborative_features = []
        for i in range(self.num_tasks):
            query = task_stack[:, i:i+1]
            
            weighted_tasks = []
            for j in range(self.num_tasks):
                weight = relation_weights[i, j]
                weighted_task = task_stack[:, j] * weight
                weighted_tasks.append(weighted_task)
            
            key_value = torch.stack(weighted_tasks, dim=1)
            
            collab_out, _ = self.task_attention(query, key_value, key_value)
            collaborative_features.append(collab_out.squeeze(1))
        
        return collaborative_features, total_score.squeeze(-1)

class SuperMultiTaskSkinNet(nn.Module):
    def __init__(self, num_modalities=5, num_tasks=6, config=None):
        super(SuperMultiTaskSkinNet, self).__init__()
        self.num_modalities = num_modalities
        self.num_tasks = num_tasks
        self.config = config or {}
        
        self.modality_backbones = nn.ModuleList()
        for i in range(num_modalities):
            # 避免在线下载预训练权重；我们会从本地 checkpoint 加载
            try:
                from torchvision.models import resnet18
                backbone = resnet18(weights=None)
            except Exception:
                backbone = models.resnet18(pretrained=False)
            backbone.layer4 = nn.Sequential(
                backbone.layer4,
                nn.AdaptiveAvgPool2d((7, 7)),
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
        
        self.super_transformer = SuperCrossImageTransformer(
            512, 8, 4, num_modalities
        )
        self.super_pyramid = SuperPyramidPooling(512)
        self.super_tsaa = SuperTSAA(512, num_tasks)
        
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
        
        modality_features = []
        for i in range(self.num_modalities):
            modal_img = images[:, i]
            feature_map = self.modality_backbones[i](modal_img)
            pooled_features = F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
            modality_features.append(pooled_features)
        
        modality_features = torch.stack(modality_features, dim=1)
        
        fused_features = self.super_transformer(modality_features)
        
        fused_feature_map = fused_features.view(batch_size, 512, 1, 1)
        fused_feature_map = F.interpolate(fused_feature_map, size=(7, 7), mode='nearest')
        enhanced_features = self.super_pyramid(fused_feature_map)
        enhanced_features = F.adaptive_avg_pool2d(enhanced_features, (1, 1)).flatten(1)
        
        task_features, total_score = self.super_tsaa(enhanced_features)
        
        main_predictions = []
        for i, task_head in enumerate(self.main_task_heads):
            main_pred = task_head(task_features[i])
            main_predictions.append(main_pred)
        main_predictions = torch.cat(main_predictions, dim=1)
        
        aux_predictions = []
        for i, aux_head in enumerate(self.auxiliary_task_heads):
            aux_pred = aux_head(enhanced_features)
            aux_predictions.append(aux_pred)
        aux_predictions = torch.cat(aux_predictions, dim=1)
        
        ensemble_predictions = 0.8 * main_predictions + 0.2 * aux_predictions
        
        return {
            'task_predictions': ensemble_predictions,
            'main_predictions': main_predictions,
            'aux_predictions': aux_predictions,
            'total_score': total_score,
            'fused_features': enhanced_features
        }

class SmartSkinAnalyzer:
    def __init__(self, model_path='models/super_multitask_skinnet.pth', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.task_names = [
            '细纹/皱纹 (Fine Lines/Wrinkles)',
            '色素沉着 (Pigmentation)',
            '毛孔 (Pores)',
            '红血丝 (Redness)',
            '紫外线损伤 (UV Damage)',
            '整体质地 (Overall Texture)'
        ]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = self._load_model(model_path)
        print("Model loaded successfully")
    
    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = SuperMultiTaskSkinNet(num_modalities=5, num_tasks=6)

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model epoch: {checkpoint.get('epoch', 'Unknown')}")
                print(f"Best accuracy: {checkpoint.get('best_acc', 'Unknown'):.4f}")
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Failed to load model weights: {e}")
            raise

        model.to(self.device)
        model.eval()
        return model
    
    def split_six_grid_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')

        width, height = image.size
        print(f"Original image size: {width} x {height}")

        grid_width = width // 3
        grid_height = height // 2

        modality_images = []

        for row in range(2):
            for col in range(3):
                left = col * grid_width
                top = row * grid_height
                right = left + grid_width
                bottom = top + grid_height

                grid_image = image.crop((left, top, right, bottom))
                modality_images.append(grid_image)
                print(f"Extracted grid [{row+1},{col+1}]: ({left},{top}) -> ({right},{bottom})")

        print(f"Successfully split image into {len(modality_images)} modalities")
        return modality_images
    
    def preprocess_image(self, image_path):
        print(f"Processing 6-grid image...")

        modality_images = self.split_six_grid_image(image_path)

        modal_tensors = []
        for i, modal_image in enumerate(modality_images):
            if i >= 5:
                break
            modal_tensor = self.transform(modal_image)
            modal_tensors.append(modal_tensor)
            print(f"Modality {i+1} preprocessed: {modal_tensor.shape}")

        while len(modal_tensors) < 5:
            modal_tensors.append(modal_tensors[-1])
            print(f"Padding modality, current count: {len(modal_tensors)}")

        images = torch.stack(modal_tensors, dim=0).unsqueeze(0)

        print(f"Image preprocessing complete: {images.shape}")
        return images.to(self.device)
    
    def predict(self, images, use_tta=False):
        with torch.no_grad():
            if use_tta:
                print("Using test-time augmentation...")
                predictions = []

                outputs = self.model(images)
                predictions.append(outputs['task_predictions'])

                flipped_images = torch.flip(images, dims=[-1])
                outputs_flip = self.model(flipped_images)
                predictions.append(outputs_flip['task_predictions'])

                avg_predictions = torch.mean(torch.stack(predictions), dim=0)

                final_outputs = {
                    'task_predictions': avg_predictions,
                    'total_score': (outputs['total_score'] + outputs_flip['total_score']) / 2
                }
            else:
                final_outputs = self.model(images)

            task_predictions = torch.clamp(final_outputs['task_predictions'], 0.0, 3.0)
            total_score = torch.clamp(final_outputs['total_score'], 0.0, 3.0)

            return {
                'task_predictions': task_predictions.cpu().numpy()[0],
                'total_score': total_score.cpu().numpy()[0] if total_score.dim() > 0 else total_score.cpu().numpy(),
                'raw_outputs': final_outputs
            }
    
    def analyze_image(self, image_path, use_tta=True, save_results=True):
        print(f"Starting 6-grid skin image analysis")

        images = self.preprocess_image(image_path)

        print(f"Running model prediction...")
        results = self.predict(images, use_tta=use_tta)

        interpretation = self.interpret_results(results)
        print("\n" + interpretation)

        if save_results:
            base_name = os.path.splitext(os.path.basename(image_path))[0] if isinstance(image_path, str) else "analysis"
            report_path = f"skin_analysis_{base_name}.txt"

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(interpretation)
            print(f"Report saved: {report_path}")

        return {
            'predictions': results,
            'interpretation': interpretation,
            'image_path': image_path
        }
    
    def interpret_results(self, results):
        task_predictions = results['task_predictions']
        total_score = results['total_score']

        report = []
        report.append("AI Skin Analysis Report")
        report.append("=" * 50)
        report.append(f"Overall Score: {total_score:.2f}/3.0")
        report.append("")

        def get_severity_level(score):
            if score <= 0.5:
                return "Excellent Q1"
            elif score <= 1.5:
                return "Good Q2"
            elif score <= 2.5:
                return "Needs Attention Q3"
            else:
                return "Needs Improvement Q4"

        report.append("Detailed Analysis:")
        for i, (task_name, score) in enumerate(zip(self.task_names, task_predictions)):
            level = get_severity_level(score)
            progress_bar = "█" * int(score * 3) + "░" * (9 - int(score * 3))
            report.append(f"   {task_name}")
            report.append(f"      Score: {score:.2f}/3.0 ({level}) [{progress_bar}]")
            report.append("")

        report.append("Personalized Care Recommendations:")

        high_score_tasks = [(i, score) for i, score in enumerate(task_predictions) if score > 2.0]
        medium_score_tasks = [(i, score) for i, score in enumerate(task_predictions) if 1.5 < score <= 2.0]
        good_tasks = [(i, score) for i, score in enumerate(task_predictions) if score <= 1.5]

        if high_score_tasks:
            report.append("   Priority Improvements:")
            task_suggestions = {
                0: "Use anti-wrinkle serum and increase moisturizing",
                1: "Use whitening products and sun protection",
                2: "Deep cleansing and pore-minimizing products",
                3: "Use anti-sensitivity products, avoid irritation",
                4: "Strengthen sun protection and use repair serum",
                5: "Use exfoliating products to improve texture"
            }
            for i, score in high_score_tasks:
                task_name = self.task_names[i].split(' (')[0]
                suggestion = task_suggestions.get(i, "Consult professional dermatologist")
                report.append(f"      - {task_name}: {suggestion}")

        if medium_score_tasks:
            report.append("   Preventive Care:")
            for i, score in medium_score_tasks:
                task_name = self.task_names[i].split(' (')[0]
                report.append(f"      - {task_name}: Continue current care, moderate enhancement")

        if good_tasks:
            report.append("   Good Condition:")
            for i, score in good_tasks:
                task_name = self.task_names[i].split(' (')[0]
                report.append(f"      - {task_name}: Excellent condition, maintain current care")

        report.append("\nOverall Care Recommendations:")
        if total_score < 0.5:
            report.append("   Skin condition is excellent! Continue regular skincare routine")
        elif total_score < 1.5:
            report.append("   Skin condition is good, recommend regular deep care")
        elif total_score < 2.5:
            report.append("   Skin needs enhanced care, recommend targeted skincare plan")
        else:
            report.append("   Skin issues are noticeable, recommend consulting professional")

        report.append("\nDaily Skincare Tips:")
        report.append("   - Use SPF30+ sunscreen daily")
        report.append("   - Maintain adequate sleep and hydration")
        report.append("   - Regularly change pillowcases and towels")
        report.append("   - Avoid over-cleansing and excessive exfoliation")

        return "\n".join(report)

def analyze_skin(image_path, model_path='models/super_multitask_skinnet.pth',
                use_tta=True, save_results=True):
    analyzer = SmartSkinAnalyzer(model_path)

    results = analyzer.analyze_image(
        image_path=image_path,
        use_tta=use_tta,
        save_results=save_results
    )

    return results


if __name__ == "__main__":
    try:
        print("=== SuperSkinGPT 6-Grid Skin Analysis System ===")
        print("Usage:")
        print("1. Single analysis: result = analyzer.analyze_image('6grid_image.jpg')")
        print("2. Quick analysis: result = analyze_skin('6grid_image.jpg')")
        print("3. Ensure input image is in 2x3 grid format")

        analyzer = SmartSkinAnalyzer('models/super_multitask_skinnet.pth')
        print("\nSystem ready for 6-grid skin image analysis")

    except FileNotFoundError:
        print("Error: Model file not found at models/super_multitask_skinnet.pth")
    except Exception as e:
        print(f"Error: Initialization failed: {e}")