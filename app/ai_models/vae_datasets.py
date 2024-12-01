from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, resize=(64, 64)):
        """
        Args:
            image_dir (str): 이미지가 저장된 폴더 경로
            transform (callable, optional): 이미지에 적용할 변환기
            resize (tuple, optional): 이미지 크기를 조정할 (width, height)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.resize = resize
        self.image_paths = list(Path(image_dir).glob("*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert("L")  # "L"로 변환하여 흑백 이미지로 로드
        
        if self.resize:
            image = image.resize(self.resize)
        
        if self.transform:
            image = self.transform(image)
            
        return image
