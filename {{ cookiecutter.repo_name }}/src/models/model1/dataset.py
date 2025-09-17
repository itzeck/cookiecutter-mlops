import torch
import torchvision.tv_tensors as tv_tensors
from PIL import Image


class WeldDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        conversion_transforms=None,
        post_transforms=None,
    ):
        self.paths = paths

        self.conversion_transforms = conversion_transforms
        self.post_transforms = post_transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path)

        if self.conversion_transforms:
            image = self.conversion_transforms(image)

        # get the json file
        mask_path = path.replace(path.split("/")[-2], "gt_masks").replace(
            ".tif", ".png"
        )

        mask = Image.open(mask_path)
        mask = tv_tensors.Mask(mask, dtype=torch.int64)
        if self.post_transforms:
            image, mask = self.post_transforms(image, mask)

        return image, mask.squeeze(0)
