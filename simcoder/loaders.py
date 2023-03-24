from pathlib import Path
from typing import Callable, Optional, Any

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, is_image_file


class UnlabelledImageFolder(VisionDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        super().__init__(root, transform=transform, target_transform=None)
        self.filepaths = [
            p for p in Path(root).glob("./*") if is_image_file(p.as_posix())
        ]

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Any:
        path = str(self.filepaths[index])
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
