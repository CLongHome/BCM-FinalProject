import os
import torch
from config import PathConfig, TrainingConfig
from data_processor import DataProcessor
from models import Generator, Discriminator
from trainer import WGANTrainer
from evaluator import Evaluator
import warnings

warnings.filterwarnings('ignore')


def main():
    # 創建輸出目錄
    os.makedirs(PathConfig.output_dir, exist_ok=True)

    # === 1. 數據處理 ===
    processor = DataProcessor()
    adata = processor.load_and_preprocess()
    adata_train, adata_test = processor.train_test_split()

    # 準備訓練數據
    X_train = adata_train.X.toarray() if hasattr(adata_train.X, "toarray") else adata_train.X

    # === 2. 模型初始化 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(
        noise_dim=TrainingConfig.noise_dim,
        hidden_dims=TrainingConfig.g_hidden,
        output_dim=X_train.shape[1]
    ).to(device)

    D = Discriminator(
        input_dim=X_train.shape[1],
        hidden_dims=TrainingConfig.d_hidden
    ).to(device)

    # === 3. 訓練 ===
    trainer = WGANTrainer(G, D, X_train, device)
    trainer.train()
    trainer.plot_loss_curve()

    # === 4. 生成樣本 ===
    X_test = adata_test.X.toarray() if hasattr(adata_test.X, "toarray") else adata_test.X
    generated_cells = trainer.generate_samples(len(X_test))

    # === 5. 評估 ===
    # 降維可視化
    Evaluator.visualize_umap(X_test, generated_cells)
    Evaluator.visualize_tsne(X_test, generated_cells)
    Evaluator.visualize_pca(X_test, generated_cells)

    # 分類評估
    X_train_original = X_train
    y_train = adata_train.obs["cell.type"]
    X_test = X_test
    y_test = adata_test.obs["cell.type"]

    print("\n=== Original Data Only ===")
    Evaluator.evaluate_classification(X_train_original, y_train, X_test, y_test)

    print("\n=== Original + Synthetic Data ===")
    Evaluator.evaluate_classification(X_train_original, y_train, X_test, y_test, generated_cells)

    # 基因表達分析
    Evaluator.analyze_gene_expression(
        X_train_original,
        generated_cells,
        gene_names=adata.var_names
    )


if __name__ == "__main__":
    main()