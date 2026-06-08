import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
#from your_dataset_file import Pollen149CSVDataset


class Pollen149CSVDataset(Dataset):
    def __init__(self, root, csv_path, transform=None, is_train=True):
        """
        - CSV 至少包含 'image' 列；
        - 若包含 'label' 列（整数0~148），则作为监督信号使用；
        - is_train 仅用于外部决定 transform（增强/不增强），与是否有标签无关。
        """
        self.root = root
        self.transform = transform
        self.is_train = is_train

        # 先读，不强制 dtype，防止无 label 时出错
        self.annotations = pd.read_csv(csv_path, header=0, encoding='utf-8')

        # 统一清洗路径列
        if 'image' not in self.annotations.columns:
            raise ValueError(f"CSV需包含'image'列：{csv_path}")
        self.annotations['image'] = self.annotations['image'].astype(str).str.strip()

        # 是否有标签列
        self.has_label = 'label' in self.annotations.columns

        if self.has_label:
            # 转 int 并校验范围
            self.annotations['label'] = pd.to_numeric(self.annotations['label'], errors='raise').astype(int)
            invalid = self.annotations[(self.annotations['label'] < 0) | (self.annotations['label'] > 148)]
            if not invalid.empty:
                raise ValueError(
                    "CSV中的label需在[0,148]范围内；示例错误：\n"
                    + invalid[['image', 'label']].head().to_string()
                )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_rel_path = row['image']
        img_path = os.path.join(self.root, img_rel_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"图像不存在：{img_path}\nCSV路径为 '{img_rel_path}'，root='{self.root}'"
            )

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"加载图像失败：{img_path}\n错误：{e}") from e

        if self.transform:
            image = self.transform(image)

        if self.has_label:
            label = torch.tensor(int(row['label']), dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)

        return image, label
