import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CPD1CSVDataset(Dataset):
    def __init__(self, root, csv_path, transform=None, is_train=True):
        """
        适配 CPD-1 (Cretan Pollen Dataset v1) 数据集（20 个类别，PNG 格式）
        - CSV 必须包含 'image' 列（存储图片相对路径，支持 .png）；
        - 若包含 'label' 列（整数0~19），则作为监督信号使用；
        - is_train 仅用于外部决定 transform（增强/不增强），与是否有标签无关。
        """
        self.root = root
        self.transform = transform
        self.is_train = is_train

        # 读取CSV
        self.annotations = pd.read_csv(csv_path, header=0, encoding='utf-8')

        # 统一清洗路径列（必须包含'image'列）
        if 'image' not in self.annotations.columns:
            raise ValueError(f"CSV需包含'image'列：{csv_path}")

        self.annotations['image'] = self.annotations['image'].astype(str).str.strip()
        self.annotations['image'] = self.annotations['image'].str.replace('\\', '/')  # 兼容Windows路径
        self.annotations['image'] = self.annotations['image'].str.replace(r'//+', '/')  # 合并多余斜杠

        # 检查是否包含标签列
        self.has_label = 'label' in self.annotations.columns

        if self.has_label:
            # 标签转int并校验范围（20类对应 0~19）
            self.annotations['label'] = pd.to_numeric(self.annotations['label'], errors='raise').astype(int)
            invalid_labels = self.annotations[(self.annotations['label'] < 0) | (self.annotations['label'] > 19)]
            if not invalid_labels.empty:
                raise ValueError(
                    f"CSV中的label需在[0,19]范围内（20个类别）；示例错误：\n"
                    + invalid_labels[['image', 'label']].head().to_string()
                )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_rel_path = row['image']
        img_path = os.path.join(self.root, img_rel_path)

        # 校验图片路径存在性
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"图像不存在：{img_path}\nCSV路径为 '{img_rel_path}'，root='{self.root}'"
            )

        # 加载 PNG 图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"加载图像失败：{img_path}\n错误：{e}") from e

        # 应用数据增强/预处理
        if self.transform:
            image = self.transform(image)

        # 处理标签
        if self.has_label:
            label = torch.tensor(int(row['label']), dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)

        return image, label
