import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class NewZealandPollenCSVDataset(Dataset):
    def __init__(self, root, csv_path, transform=None, is_train=True):
        """
        新西兰花粉（New Zealand Pollen）数据集 CSV 读取类
        - CSV 至少包含 'image' 列（图像相对路径）；
        - 若包含 'label' 列（整数，范围需根据实际类别数修改），则作为监督信号使用；
        - is_train 仅用于外部决定 transform（增强/不增强），与是否有标签无关。
        
        【重要】请根据实际数据集修改 label 校验范围（如下示例为0~27，对应28个类别）
        """
        self.root = root
        self.transform = transform
        self.is_train = is_train

        # 读取CSV，不强制dtype，避免无label列时出错（添加编码容错备注）
        try:
            self.annotations = pd.read_csv(csv_path, header=0, encoding='utf-8')
        except UnicodeDecodeError:
            # 兼容中文/特殊字符的CSV（若有）
            self.annotations = pd.read_csv(csv_path, header=0, encoding='gbk')

        # 统一清洗图像路径列
        if 'image' not in self.annotations.columns:
            raise ValueError(f"【新西兰花粉数据集】CSV文件必须包含'image'列，请检查：{csv_path}")
        # 清洗：去空格 + 空值过滤
        self.annotations['image'] = self.annotations['image'].astype(str).str.strip()
        # 过滤空路径行（避免拼接出无效路径）
        empty_path_rows = self.annotations[self.annotations['image'] == '']
        if not empty_path_rows.empty:
            raise ValueError(
                f"【新西兰花粉数据集】CSV中存在空的image路径，错误行索引：\n"
                + empty_path_rows.index.tolist()[:5]  # 只显示前5行
            )

        # 标记是否包含标签列
        self.has_label = 'label' in self.annotations.columns

        if self.has_label:
            # ========== 需修改处 ==========
            # 请将 0~27 改为你数据集实际的 label 范围（例如0~15对应16个类别）
            LABEL_MIN = 0
            LABEL_MAX = 45  # 示例：28个类别
            # =============================
            
            # 转换标签为整数并校验范围（添加数据集标识，方便排查）
            self.annotations['label'] = pd.to_numeric(self.annotations['label'], errors='raise').astype(int)
            invalid_labels = self.annotations[(self.annotations['label'] < LABEL_MIN) | (self.annotations['label'] > LABEL_MAX)]
            if not invalid_labels.empty:
                raise ValueError(
                    f"【新西兰花粉数据集】CSV中的label值必须在[{LABEL_MIN},{LABEL_MAX}]范围内（请检查LABEL_MIN/LABEL_MAX是否配置正确）；错误示例：\n"
                    + invalid_labels[['image', 'label']].head().to_string()
                )

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.annotations)

    def __getitem__(self, idx):
        """读取单张图像和对应标签（无标签时返回-1）"""
        row = self.annotations.iloc[idx]
        img_rel_path = row['image']
        
        # ========== 新增：路径兼容修复（核心） ==========
        # 1. 替换Windows反斜杠为Linux正斜杠 2. 再次去空格（双重保险）
        img_rel_path = img_rel_path.replace('\\', '/').strip()
        # ==============================================

        img_abs_path = os.path.join(self.root, img_rel_path)

        # 校验图像文件是否存在（添加数据集标识，方便定位）
        if not os.path.exists(img_abs_path):
            raise FileNotFoundError(
                f"【新西兰花粉数据集】图像文件不存在：{img_abs_path}\n"
                f"CSV中记录的相对路径：'{row['image']}'，转换后路径：'{img_rel_path}'，数据集根目录：'{self.root}'"
            )

        # 读取图像（强制转为RGB，避免灰度图/透明通道问题）
        try:
            image = Image.open(img_abs_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"【新西兰花粉数据集】加载图像失败：{img_abs_path}\n具体错误：{str(e)}") from e

        # 应用数据增强/归一化
        if self.transform is not None:
            image = self.transform(image)

        # 处理标签（有标签返回对应值，无标签返回-1）
        if self.has_label:
            label = torch.tensor(int(row['label']), dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)

        return image, label
