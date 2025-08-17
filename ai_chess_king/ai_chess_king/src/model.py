"""
model.py
========

定義序列式 LSTM 生成器與對抗式判別器，並簡易封裝 GAN 訓練所需方法。
本模組使用 PyTorch 實作。

生成器 (Generator) 以一段棋招序列為條件，輸出下一步的機率分布。
判別器 (Discriminator) 以完整棋譜序列為輸入，判斷其是否來自真實資料。

為了保留示例簡潔性，這裡僅實作單棋種的簡易結構；實際使用時可依棋盤尺寸
調整輸入維度與輸出空間。
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    # 如果未安裝 PyTorch，定義佔位符類別，以避免導入錯誤。
    torch = None  # type: ignore
    class _Dummy:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch 未安裝，無法使用模型")
    nn = _Dummy()  # type: ignore
    F = _Dummy()  # type: ignore


class LSTMGenerator(nn.Module):
    """簡易 LSTM 生成器。

    參數
    ------
    vocab_size: 總招法數量（字典大小）。
    embed_dim: 嵌入層維度。
    hidden_dim: LSTM 隱藏層大小。
    num_layers: LSTM 層數。
    max_length: 輸入序列最大長度。
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2, max_length: int = 40) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size + 1)  # +1 for PAD/UNKNOWN
        self.max_length = max_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len)
        emb = self.embed(x)
        output, _ = self.lstm(emb)
        # 取最後一個時間步的輸出
        last_output = output[:, -1, :]
        logits = self.fc(last_output)
        return logits

    def generate(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """根據輸入序列生成下一步的索引。"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits / temperature, dim=-1)
            next_move = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_move


class LSTMDiscriminator(nn.Module):
    """簡易 LSTM 判別器。輸入完整對局序列，輸出是否真實的概率。"""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2, max_length: int = 40) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        output, _ = self.lstm(emb)
        last_output = output[:, -1, :]
        logits = self.fc(last_output)
        prob = torch.sigmoid(logits)
        return prob


def gan_loss(discriminator: LSTMDiscriminator, real_samples: torch.Tensor, fake_samples: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """計算 GAN 判別器和生成器的損失。

    參數
    ------
    discriminator: 判別器模型
    real_samples: 真實序列張量 (batch, seq_len)
    fake_samples: 生成器產生的序列張量 (batch, seq_len)
    device: 所在裝置

    回傳
    ------
    d_loss: 判別器損失
    g_loss: 生成器損失
    """
    criterion = nn.BCELoss()
    # 判別真樣本為 1
    real_labels = torch.ones((real_samples.size(0), 1), device=device)
    # 判別假樣本為 0
    fake_labels = torch.zeros((fake_samples.size(0), 1), device=device)
    # 判別器對真樣本的預測
    d_real = discriminator(real_samples)
    d_real_loss = criterion(d_real, real_labels)
    # 判別器對假樣本的預測
    d_fake = discriminator(fake_samples.detach())
    d_fake_loss = criterion(d_fake, fake_labels)
    d_loss = d_real_loss + d_fake_loss
    # 生成器試圖騙過判別器，目標標籤為 1
    g_pred = discriminator(fake_samples)
    g_loss = criterion(g_pred, real_labels)
    return d_loss, g_loss
