from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 무GUI 백엔드 설정
import matplotlib.pyplot as plt

from app.core.config import settings
from ebrandom.linear_congruential import random_list as lc_random_list
from ebrandom.mersenne_twister import random_list as mt_random_list
from ebrandom.mid_square import random_list as ms_random_list
from app.api.rng.box_muller.box_muller import generate_latent_samples  # Box-Muller 변환 함수 가져오기

router = APIRouter()

# 알고리즘별 난수 생성기 매핑
RANDOM_FUNCTIONS = {
    "lc": lc_random_list,  # Linear Congruential
    "mt": mt_random_list,  # Mersenne Twister
    "ms": ms_random_list,  # Mid Square
}

# 그래프 저장 디렉토리
GRAPH_DIR = settings.distribution_graph_dir
os.makedirs(GRAPH_DIR, exist_ok=True)


@router.get("/distribution_map/")
def distribution_map(
    algorithm: str = Query(..., regex="^(lc|mt|ms)$"),  # 알고리즘 이름 (lc, mt, ms)
    distribution_type: str = Query(..., regex="^(un|nm)$")  # 분포 유형 (uniform, normal)
):
    """
    알고리즘과 분포 유형에 따라 분포도 그래프 생성 및 반환
    """
    graph_file = os.path.join(GRAPH_DIR, f"{algorithm}_{distribution_type}.png")

    # 알고리즘 선택
    if algorithm not in RANDOM_FUNCTIONS:
        raise HTTPException(status_code=400, detail="Invalid algorithm name.")
    random_list = RANDOM_FUNCTIONS[algorithm]

    # 분포 데이터 생성
    if distribution_type == "un":
        data = np.array(random_list(0, 100, 1000))  # 균등 분포 생성
        title = f"{algorithm.upper()} Uniform Distribution"
    elif distribution_type == "nm":
        # Box-Muller 변환을 통해 정규 분포 생성
        data = generate_latent_samples(num_samples=1000, latent_dim=2, algorithm=algorithm).flatten()
        title = f"{algorithm.upper()} Normal Distribution"
    else:
        raise HTTPException(status_code=400, detail="Invalid distribution type.")

    # 분포도 그래프 생성 및 저장
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(graph_file)
    plt.close()

    # 생성된 그래프 파일 반환
    return FileResponse(graph_file)
