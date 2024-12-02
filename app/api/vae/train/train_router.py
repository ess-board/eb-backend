from fastapi import APIRouter
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
import os

from app.ai_models.vae_model import VAE, vae_loss
from app.ai_models.vae_datasets import CustomImageDataset
from app.core.config import settings
from app.api.vae.schemas import ResponseModel

router = APIRouter()

@router.post("/train/", response_model=ResponseModel)
async def train_vae(process_id: str, epochs: int = 10):
    EXTRACT_DIR = settings.extracted_dataset_dir
    SAVE_MODEL_DIR = settings.save_model_dir
    
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

    folder_path = os.path.join(EXTRACT_DIR, process_id)

    # 데이터셋
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = CustomImageDataset(folder_path, transform=transform, resize=(64, 64))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = VAE()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습
    num_epochs = epochs
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"에포크 [{epoch+1}/{num_epochs}], 손실: {total_loss / len(dataloader.dataset):.4f}")

    # 모델 저장
    model_path = os.path.join(SAVE_MODEL_DIR, f"vae_{process_id}.pt")
    torch.save(model.state_dict(), model_path)

    return {
        "status": "success",
        "message": "모델이 성공적으로 학습되었습니다.",
        "process_id": process_id,
    }
