from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routers

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

routers.include_all_routers(app)