import numpy as np
from ebrandom.linear_congruential import random01 as lc_random01
from ebrandom.mersenne_twister import random01 as mt_random01
from ebrandom.mid_square import random01 as ms_random01

# 알고리즘 약자와 난수 생성기 매핑
RANDOM_FUNCTIONS = {
    "lc": lc_random01,  # Linear Congruential
    "mt": mt_random01,  # Mersenne Twister
    "ms": ms_random01,  # Mid Square
}

def generate_latent_samples(num_samples, latent_dim, algorithm="lc"):
    if algorithm not in RANDOM_FUNCTIONS:
        raise ValueError(f"알 수 없는 알고리즘: {algorithm}. 사용 가능한 알고리즘: {list(RANDOM_FUNCTIONS.keys())}")

    random01 = RANDOM_FUNCTIONS[algorithm]

    total_size = num_samples * latent_dim
    if total_size % 2 != 0:
        raise ValueError("num_samples * latent_dim must be even for reshaping.")

    u1 = np.array([random01() for _ in range(total_size // 2)]).reshape(num_samples, latent_dim // 2)
    u2 = np.array([random01() for _ in range(total_size // 2)]).reshape(num_samples, latent_dim // 2)

    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)

    z = np.hstack((z0, z1))[:, :latent_dim]
    return z
