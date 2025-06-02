import torch
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from config import TrainingConfig


class WGANTrainer:
    def __init__(self, G, D, train_data, device=None):
        self.G = G
        self.D = D
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 準備數據加載器
        X_tensor = torch.tensor(train_data, dtype=torch.float32)
        self.train_loader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=TrainingConfig.batch_size,
            shuffle=True,
            drop_last=True
        )

        # 優化器
        self.optimizer_G = torch.optim.Adam(
            G.parameters(),
            lr=TrainingConfig.lr,
            betas=TrainingConfig.betas
        )
        self.optimizer_D = torch.optim.Adam(
            D.parameters(),
            lr=TrainingConfig.lr,
            betas=TrainingConfig.betas
        )

        # 訓練記錄
        self.g_losses = []
        self.d_losses = []

    def gradient_penalty(self, real_samples, fake_samples):
        """計算gradient penalty"""
        batch_size = real_samples.size(0)
        epsilon = torch.rand(batch_size, 1, device=self.device).expand_as(real_samples)
        interpolates = epsilon * real_samples + (1 - epsilon) * fake_samples
        interpolates.requires_grad_(True)

        d_interpolates = self.D(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return ((slopes - 1) ** 2).mean()

    def train_epoch(self, epoch, num_epochs):
        """單一epoch訓練"""
        self.G.train()
        self.D.train()
        start_time = time.time()

        g_loss_sum = 0.0
        d_loss_sum = 0.0
        num_batches = len(self.train_loader)

        for i, (real_samples,) in enumerate(self.train_loader):
            real_samples = real_samples.to(self.device)

            # 訓練判別器
            for _ in range(TrainingConfig.d_iters_per_g):
                z = torch.randn(real_samples.size(0), TrainingConfig.noise_dim, device=self.device)
                with torch.no_grad():
                    fake_samples = self.G(z).detach()

                d_real = self.D(real_samples).mean()
                d_fake = self.D(fake_samples).mean()
                gp = self.gradient_penalty(real_samples, fake_samples)
                d_loss = d_fake - d_real + TrainingConfig.lambda_gp * gp

                self.optimizer_D.zero_grad()
                d_loss.backward()
                self.optimizer_D.step()

            # 訓練生成器
            z = torch.randn(real_samples.size(0), TrainingConfig.noise_dim, device=self.device)
            fake_samples = self.G(z)
            g_loss = -self.D(fake_samples).mean()

            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()

            # 記錄損失
            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()

            # 打印訓練進度
            if i % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] Batch [{i}/{num_batches}] "
                    f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} "
                    f"Time: {time.time() - start_time:.2f}s"
                )
                start_time = time.time()

        # 記錄epoch平均損失
        self.g_losses.append(g_loss_sum / num_batches)
        self.d_losses.append(d_loss_sum / num_batches)

    def train(self):
        """完整訓練流程"""
        for epoch in range(TrainingConfig.num_epochs):
            self.train_epoch(epoch, TrainingConfig.num_epochs)

    def generate_samples(self, num_samples, batch_size=32):
        """生成樣本"""
        self.G.eval()
        generated = []
        with torch.no_grad():
            num_batches = (num_samples + batch_size - 1) // batch_size
            for _ in range(num_batches):
                z = torch.randn(batch_size, TrainingConfig.noise_dim, device=self.device)
                gen = self.G(z).cpu().numpy()
                generated.append(gen)
        return np.vstack(generated)[:num_samples]

    def plot_loss_curve(self):
        """畫出訓練過程中生成器和判別器的損失曲線"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.g_losses) + 1), self.g_losses, label="Generator Loss")
        plt.plot(range(1, len(self.d_losses) + 1), self.d_losses, label="Critic Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("WGAN Training Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()
