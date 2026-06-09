import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class NewZealandPollenCSVDataset(Dataset):
    def __init__(self, root, csv_path, transform=None, is_train=True):
        """
        CSV-based dataset loader for the New Zealand Pollen Dataset.

        Expected CSV format:
            image,label

        Required columns:
            image: relative image path under the dataset root.

        Optional columns:
            label: integer class index. For the New Zealand Pollen Dataset,
                   the expected label range is [0, 45] for 46 classes.

        Notes:
            - If the label column is available, it is used as the supervised target.
            - If the label column is absent, the dataset returns -1 as a placeholder label.
            - The is_train argument is kept for compatibility with external data-building code.
              Data augmentation should be controlled by the transform argument.
        """
        self.root = root
        self.transform = transform
        self.is_train = is_train

        try:
            self.annotations = pd.read_csv(csv_path, header=0, encoding="utf-8")
        except UnicodeDecodeError:
            self.annotations = pd.read_csv(csv_path, header=0, encoding="gbk")

        if "image" not in self.annotations.columns:
            raise ValueError(f"The CSV file must contain an 'image' column: {csv_path}")

        self.annotations["image"] = self.annotations["image"].astype(str).str.strip()

        empty_path_rows = self.annotations[self.annotations["image"] == ""]
        if not empty_path_rows.empty:
            raise ValueError(
                "The CSV file contains empty image paths. "
                f"Examples of row indices: {empty_path_rows.index.tolist()[:5]}"
            )

        self.has_label = "label" in self.annotations.columns

        if self.has_label:
            label_min = 0
            label_max = 45

            self.annotations["label"] = pd.to_numeric(
                self.annotations["label"], errors="raise"
            ).astype(int)

            invalid_labels = self.annotations[
                (self.annotations["label"] < label_min) |
                (self.annotations["label"] > label_max)
            ]

            if not invalid_labels.empty:
                raise ValueError(
                    f"Labels in the CSV file must be within [{label_min}, {label_max}]. "
                    "Examples of invalid rows:\n"
                    + invalid_labels[["image", "label"]].head().to_string()
                )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """Load one image and its label. Return -1 as the label when labels are unavailable."""
        row = self.annotations.iloc[idx]

        img_rel_path = row["image"]
        img_rel_path = img_rel_path.replace("\\", "/").strip()

        img_abs_path = os.path.join(self.root, img_rel_path)

        if not os.path.exists(img_abs_path):
            raise FileNotFoundError(
                f"Image file not found: {img_abs_path}\n"
                f"Original CSV image path: '{row['image']}'\n"
                f"Normalized image path: '{img_rel_path}'\n"
                f"Dataset root: '{self.root}'"
            )

        try:
            image = Image.open(img_abs_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image: {img_abs_path}\n"
                f"Original error: {e}"
            ) from e

        if self.transform is not None:
            image = self.transform(image)

        if self.has_label:
            label = torch.tensor(int(row["label"]), dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)

        return image, label
