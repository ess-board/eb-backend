from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, resize=(64, 64), max_images=None):
        """
        Args:
            image_dir (str): 이미지가 저장된 폴더 경로
            transform (callable, optional): 이미지에 적용할 변환기
            resize (tuple, optional): 이미지 크기를 조정할 (width, height)
            max_images (int, optional): 불러올 최대 이미지 개수
        """
        self.image_dir = image_dir
        self.transform = transform
        self.resize = resize
        # .png, .jpg, .jpeg 파일 로드
        self.image_paths = list(Path(image_dir).rglob("*.[pP][nN][gG]")) + \
                           list(Path(image_dir).rglob("*.[jJ][pP][gG]")) + \
                           list(Path(image_dir).rglob("*.[jJ][pP][eE][gG]"))
        
        # 최대 개수 제한
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

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
