import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def list_images(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)]
    files.sort()
    return files

class PairedImageFolder(Dataset):
    """
    Paired dataset: low/ and high/ must contain matching filenames.
    Returns (low, high) tensors in [-1, 1].
    """
    def __init__(self, root, split="train", image_size=256, random_flip=True):
        self.low_dir = os.path.join(root, split, "low")
        self.high_dir = os.path.join(root, split, "high")
        assert os.path.isdir(self.low_dir), f"Missing: {self.low_dir}"
        assert os.path.isdir(self.high_dir), f"Missing: {self.high_dir}"

        self.files = list_images(self.low_dir)
        assert len(self.files) > 0, "No images found."
        # verify match exists
        for f in self.files[:10]:
            assert os.path.exists(os.path.join(self.high_dir, f)), f"Missing GT for {f}"

        tf = []
        tf.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC))
        if random_flip and split == "train":
            tf.append(T.RandomHorizontalFlip(p=0.5))
        tf.append(T.ToTensor())  # [0,1]
        self.to_tensor = T.Compose(tf)

        # map to [-1,1]
        self.to_m11 = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        low = Image.open(os.path.join(self.low_dir, fn)).convert("RGB")
        high = Image.open(os.path.join(self.high_dir, fn)).convert("RGB")

        low = self.to_m11(self.to_tensor(low))
        high = self.to_m11(self.to_tensor(high))
        return low, high, fn