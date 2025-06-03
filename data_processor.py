# data_processor.py
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from config import SEED, PathConfig


class DataProcessor:
    def __init__(self):
        self.adata = None

    def load_and_preprocess(self, top_var_ratio=0.2):
        """載入並預處理數據，並挑選前 top_var_ratio 變異基因"""
        # 讀取表達矩陣與註解
        expr = pd.read_csv(PathConfig.expr_path, sep="\t")
        annotation = pd.read_csv(PathConfig.annotation_path)

        # 合併與清理
        expr = expr.merge(annotation, on="gene.id").drop(columns=["gene.id"])
        expr = expr.set_index("symbol")
        expr = expr[~expr.index.duplicated(keep="first")].T

        # 建立 AnnData
        self.adata = sc.AnnData(expr)
        sc.pp.log1p(self.adata)

        # 挑選變異度前 top_var_ratio 的基因
        sc.pp.highly_variable_genes(self.adata, flavor="seurat", n_top_genes=int(self.adata.shape[1] * top_var_ratio))
        self.adata = self.adata[:, self.adata.var.highly_variable].copy()

        # 處理 metadata
        meta = pd.read_csv(PathConfig.metadata_path, sep="\t")
        meta["sample_id"] = meta["Sample.name"].str.replace(" ", "_")
        meta = meta.set_index("sample_id").loc[self.adata.obs_names]

        # 擷取 T2D/Non T2D 狀態
        meta["T2D_status"] = meta["condition"].str.extract(r'(T2D|Non T2D)', expand=False)
        self.adata.obs = meta

        # 移除 contaminated 細胞
        self.adata = self.adata[~self.adata.obs["cell.type"].str.contains("contaminated", case=False, na=False)]

        # 清理缺失資料並設定標籤欄位
        self.adata = self.adata[~self.adata.obs["condition"].isna()]
        self.adata.obs["T2D_status"] = self.adata.obs["condition"]

        return self.adata

    def train_test_split(self, test_size=0.2):
        """拆分訓練測試集（以 T2D 狀態分類）"""
        train_idx, test_idx = train_test_split(
            range(self.adata.n_obs),
            test_size=test_size,
            stratify=self.adata.obs["T2D_status"],
            random_state=SEED
        )
        return self.adata[train_idx].copy(), self.adata[test_idx].copy()