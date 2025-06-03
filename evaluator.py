# evaluator.py
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class Evaluator:
    @staticmethod
    def visualize_umap(real_data, fake_data):
        combined = np.vstack([real_data, fake_data])
        labels = np.array(["Real"] * len(real_data) + ["Generated"] * len(fake_data))

        # UMAP
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(combined)
        sil_score = silhouette_score(embedding, (labels == "Generated").astype(int))

        plt.figure(figsize=(8, 6))
        for label in ["Real", "Generated"]:
            mask = labels == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1], label=label, alpha=0.6)
        plt.title(f"UMAP Projection\nSilhouette Score: {sil_score:.4f}")
        plt.legend()
        plt.show()

        return sil_score

    @staticmethod
    def visualize_tsne(real_data, fake_data, perplexity=30):
        combined = np.vstack([real_data, fake_data])
        labels = np.array(["Real"] * len(real_data) + ["Generated"] * len(fake_data))
        labels_numeric = (labels == "Generated").astype(int)

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_embedding = tsne.fit_transform(combined)
        sil_score = silhouette_score(tsne_embedding, labels_numeric)

        plt.figure(figsize=(8, 6))
        for label in ["Real", "Generated"]:
            mask = labels == label
            plt.scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1],
                        alpha=0.6, label=label)
        plt.title(f"t-SNE Projection\nSilhouette Score: {sil_score:.4f}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return sil_score

    @staticmethod
    def visualize_pca(real_data, fake_data):
        combined = np.vstack([real_data, fake_data])
        labels = np.array(["Real"] * len(real_data) + ["Generated"] * len(fake_data))

        pca = PCA(n_components=2)
        pca_embedding = pca.fit_transform(combined)
        sil_score = silhouette_score(pca_embedding, (labels == "Generated").astype(int))

        plt.figure(figsize=(8, 6))
        for label in ["Real", "Generated"]:
            mask = labels == label
            plt.scatter(pca_embedding[mask, 0], pca_embedding[mask, 1], label=label, alpha=0.6)
        plt.title(f"PCA Projection\nSilhouette Score: {sil_score:.4f}")
        plt.legend()
        plt.show()

        return sil_score

    @staticmethod
    def evaluate_classification(X_train_real, y_train, X_test, y_test, X_synthetic=None, y_synthetic=None):
        if X_synthetic is not None:
            X_train = np.vstack([X_train_real, X_synthetic])
            if y_synthetic is not None:
                y_train_combined = np.concatenate([y_train, y_synthetic])
            else:
                # 如果沒提供 y_synthetic，假設合成標籤跟原標籤一樣（複製前面幾筆）
                y_synthetic = np.array([y_train[0]] * len(X_synthetic))
                y_train_combined = np.concatenate([y_train, y_synthetic])
        else:
            X_train = X_train_real
            y_train_combined = y_train

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train_combined)
        y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()

    @staticmethod
    def analyze_gene_expression(real_data, synthetic_data, gene_names, top_n=20):
        """基因表達特徵分析"""
        real_var = np.var(real_data, axis=0)
        synth_var = np.var(synthetic_data, axis=0)

        # 找出高變異基因
        real_top_idx = np.argsort(real_var)[-top_n:][::-1]
        synth_top_idx = np.argsort(synth_var)[-top_n:][::-1]

        print("Top variable genes in real data:")
        print(gene_names[real_top_idx])

        print("\nTop variable genes in synthetic data:")
        print(gene_names[synth_top_idx])

        # 計算重疊率
        overlap = len(set(real_top_idx) & set(synth_top_idx))
        print(f"\nOverlap ratio: {overlap / top_n:.2%}")

        # 繪製變異數比較
        plt.figure(figsize=(10, 6))
        plt.scatter(np.log(real_var + 1e-6), np.log(synth_var + 1e-6), alpha=0.5)
        plt.xlabel("Log Variance (Real)")
        plt.ylabel("Log Variance (Synthetic)")
        plt.title("Gene Expression Variance Comparison")
        plt.plot([-15, 15], [-15, 15], 'r--')
        plt.show()