import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Pollen149CSVDataset(Dataset):
    def __init__(self, root, csv_path, transform=None, is_train=True):
        """
        CSV-based dataset loader for Pollen149.

        Expected CSV format:
            image,label

        Required columns:
            image: relative image path under the dataset root.

        Optional columns:
            label: integer class index in [0, 148].

        Notes:
            - If the label column is available, it is used as the supervised target.
            - If the label column is absent, the dataset returns -1 as a placeholder label.
            - The is_train argument is kept for compatibility with external data-building code.
              Data augmentation should be controlled by the transform argument.
        """
        self.root = root
        self.transform = transform
        self.is_train = is_train

        self.annotations = pd.read_csv(csv_path, header=0, encoding="utf-8")

        if "image" not in self.annotations.columns:
            raise ValueError(f"The CSV file must contain an 'image' column: {csv_path}")

        self.annotations["image"] = self.annotations["image"].astype(str).str.strip()

        self.has_label = "label" in self.annotations.columns

        if self.has_label:
            self.annotations["label"] = pd.to_numeric(
                self.annotations["label"], errors="raise"
            ).astype(int)

            invalid = self.annotations[
                (self.annotations["label"] < 0) |
                (self.annotations["label"] > 148)
            ]

            if not invalid.empty:
                raise ValueError(
                    "Labels in the CSV file must be within [0, 148]. "
                    "Examples of invalid rows:\n"
                    + invalid[["image", "label"]].head().to_string()
                )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        img_rel_path = row["image"]
        img_path = os.path.join(self.root, img_rel_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image file not found: {img_path}\n"
                f"CSV image path: '{img_rel_path}', dataset root: '{self.root}'"
            )

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image: {img_path}\n"
                f"Original error: {e}"
            ) from e

        if self.transform is not None:
            image = self.transform(image)

        if self.has_label:
            label = torch.tensor(int(row["label"]), dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)

        return image, label
