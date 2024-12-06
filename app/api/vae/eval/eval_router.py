from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import JSONResponse, FileResponse
import torch
import os
from PIL import Image
import numpy as np

from app.ai_models.vae_model import VAE
from app.api.vae.schemas import ResponseModel
from app.core.config import settings
from app.api.rng.box_muller.box_muller import generate_latent_samples


router = APIRouter()

OUTPUT_DIR = settings.generated_images_dir


@router.post("/", response_model=ResponseModel)
async def evaluate_vae(
    process_id: str, 
    num_images: int = 1,
    algorithm: str = Query("mt", regex="^(lc|mt|ms)$")):
    SAVE_MODEL_DIR = settings.save_model_dir

    model_path = os.path.join(SAVE_MODEL_DIR, f"vae_{process_id}.pt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found.")

    model = VAE()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    generated_images = []
    with torch.no_grad():
        for i in range(num_images):
            # 잠재 공간에서 샘플링
            latent_sample = generate_latent_samples(1, 20, algorithm=algorithm)
            latent_sample_tensor = torch.tensor(latent_sample, dtype=torch.float32)
            
            # 이미지 생성
            generated_img_flat = model.decode(latent_sample_tensor)
            generated_img = generated_img_flat.view(64, 64)
            generated_img = generated_img.clamp(0, 1)
            generated_img_np = generated_img.cpu().numpy()
            generated_img_np = (generated_img_np * 255).astype(np.uint8)
            generated_img_pil = Image.fromarray(generated_img_np, mode='L')

            # 이미지 저장
            image_name = f"generated_{process_id}_{i}.png"
            image_path = os.path.join(OUTPUT_DIR, image_name)
            generated_img_pil.save(image_path)

            # URL로 변환
            image_url = f"/api/{image_name}"
            generated_images.append(image_url)

    return JSONResponse(content={
        "status": "success",
        "message": f"{num_images}개의 이미지가 생성되었습니다.",
        "process_id": process_id
    })


@router.get("/get_image/{process_id}/{image_idx}")
async def get_image(
    process_id: str = Path(..., regex="^[a-zA-Z0-9_-]+$"),
    image_idx: int = Path(..., ge=0)
):

    filename = f"generated_{process_id}_{image_idx}.png"

    image_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(image_path)