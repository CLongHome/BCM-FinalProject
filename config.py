# config.py
import torch
import random
import numpy as np

# 固定隨機種子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 訓練參數
class TrainingConfig:
    noise_dim = 10
    g_hidden = [512, 256]  # Generator隱藏層
    d_hidden = [512, 256]  # Discriminator隱藏層
    batch_size = 32
    lr = 5e-5
    betas = (0.5, 0.9)
    lambda_gp = 5
    d_iters_per_g = 5
    num_epochs = 100

# 路徑配置
class PathConfig:
    expr_path = "dataset/GSE81608_human_islets_rpkm.txt.gz"
    annotation_path = "dataset/human_gene_annotation.csv"
    metadata_path = "dataset/human_islet_cell_identity.txt"
    output_dir = "./outputs/"