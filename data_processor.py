import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from config import SEED, PathConfig


class DataProcessor:
    def __init__(self):
        self.adata = None

    def load_and_preprocess(self):
        """載入並預處理數據"""
        # 讀取表達矩陣與註解
        expr = pd.read_csv(PathConfig.expr_path, sep="\t")
        annotation = pd.read_csv(PathConfig.annotation_path)

        # 合併與清理
        expr = expr.merge(annotation, on="gene.id").drop(columns=["gene.id"])
        expr = expr.set_index("symbol")
        expr = expr[~expr.index.duplicated(keep="first")].T

        # 建立AnnData
        self.adata = sc.AnnData(expr)
        sc.pp.log1p(self.adata)

        # 處理metadata
        meta = pd.read_csv(PathConfig.metadata_path, sep="\t")
        meta["sample_id"] = meta["Sample.name"].str.replace(" ", "_")
        meta = meta.set_index("sample_id").loc[self.adata.obs_names]
        self.adata.obs = meta

        return self.adata

    def train_test_split(self, test_size=0.2):
        """拆分訓練測試集"""
        train_idx, test_idx = train_test_split(
            range(self.adata.n_obs),
            test_size=test_size,
            stratify=self.adata.obs["cell.type"],
            random_state=SEED
        )
        return self.adata[train_idx].copy(), self.adata[test_idx].copy()