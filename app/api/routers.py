from fastapi import FastAPI, APIRouter
import app.main as main

from app.api.vae.dataset import dataset_router
from app.api.vae.train import train_router
from app.api.vae.eval import eval_router

from app.api.rng.distribution_map import distribution_map_router

vae_routers = APIRouter()
vae_routers.include_router(dataset_router.router, prefix="/dataset")
vae_routers.include_router(train_router.router, prefix="/train")
vae_routers.include_router(eval_router.router, prefix="/eval")

rng_routers = APIRouter()
rng_routers.include_router(distribution_map_router.router, prefix="/distribution_map")

def include_all_routers(app: FastAPI) -> None:
    app.include_router(vae_routers, prefix="/api/vae", tags=["VAE pipelines"])
    app.include_router(rng_routers, prefix="/api/rng", tags=["Random Number Generators"])