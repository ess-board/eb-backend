from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os


class Settings(BaseSettings):
    app_name: str

    uploaded_dataset_dir: str
    extracted_dataset_dir: str
    save_model_dir: str
    generated_images_dir: str

    class Config:
        env_file = ".env"


load_dotenv()
settings = Settings()

print(settings)
