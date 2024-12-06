from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import zipfile
from pathlib import Path
import os
import cuid

from app.core.config import settings
from app.api.vae.schemas import ResponseModel

router = APIRouter()


@router.post("/upload_zip/", response_model=ResponseModel)
async def upload_zip(zip_file: UploadFile = File(...)):
    UPLOAD_DIR = settings.uploaded_dataset_dir
    EXTRACT_DIR = settings.extracted_dataset_dir

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    process_id = cuid.cuid()

    unique_filename = f"{process_id}.zip"
    zip_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(zip_file.file, buffer)

    unique_extract_folder = os.path.join(EXTRACT_DIR, process_id)
    os.makedirs(unique_extract_folder, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unique_extract_folder)

    # Flatten folder structure
    for dir_path in Path(unique_extract_folder).rglob('*'):
        if dir_path.is_dir():
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    shutil.move(str(file_path), unique_extract_folder)
            shutil.rmtree(dir_path)

    # Delete non-PNG, JPG, JPEG files
    for file_path in Path(unique_extract_folder).rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            file_path.unlink()  # Delete the file

    return {
        "status": "success",
        "message": "이미지가 성공적으로 업로드되고 압축이 해제되었습니다. PNG, JPG, JPEG 이외의 파일은 삭제되었습니다.",
        "process_id": process_id
    }
