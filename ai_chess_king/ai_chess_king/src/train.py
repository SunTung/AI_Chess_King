"""
train.py
========

模型訓練腳本。可指定棋種、訓練週期、批次大小以及是否使用 GAN 進行對抗式訓練。
目前示例主要以西洋棋 PGN 資料作為示範，其他棋種需自行撰寫解析。
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import process_chess_pgn, build_vocab, encode_moves
from model import LSTMGenerator, LSTMDiscriminator, gan_loss


def prepare_dataset(game: str, data_dir: str, max_length: int) -> Tuple[np.ndarray, dict]:
    """根據棋種讀取並返回訓練資料和字典。"""
    if game == "chess":
        # 假設 data_dir 下有 chess.pgn
        pgn_path = os.path.join(data_dir, "chess.pgn")
        data, vocab = process_chess_pgn(pgn_path, max_games=1000, max_length=max_length)
        return data, vocab
    else:
        raise NotImplementedError(f"暫未支援 {game} 的資料處理")


def train_generator_only(args) -> None:
    """僅用交叉熵訓練生成器，以預測下一步。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, vocab = prepare_dataset(args.game, args.data_dir, args.max_length)
    if len(data) == 0:
        print("沒有可用資料，無法訓練。")
        return
    vocab_size = len(vocab)
    # 構造訓練樣本：輸入序列為前 n-1 手，標籤為第 n 手
    inputs = data[:, :-1]
    targets = data[:, -1]
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = LSTMGenerator(vocab_size=vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, max_length=args.max_length)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}")

    # 儲存模型
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{args.game}_generator.pt")
    torch.save({"model_state": model.state_dict(), "vocab": vocab}, model_path)
    print(f"模型已儲存至 {model_path}")


def train_with_gan(args) -> None:
    """簡易對抗訓練：生成器與判別器交替更新。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, vocab = prepare_dataset(args.game, args.data_dir, args.max_length)
    if len(data) == 0:
        print("沒有可用資料，無法訓練。")
        return
    vocab_size = len(vocab)
    dataset = TensorDataset(torch.tensor(data, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = LSTMGenerator(vocab_size=vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, max_length=args.max_length)
    discriminator = LSTMDiscriminator(vocab_size=vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, max_length=args.max_length)
    generator.to(device)
    discriminator.to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)
            # 生成假樣本：將真樣本序列中的最後一步替換為生成器輸出的下一步
            # 輸入為前 n-1 手
            input_seq = real_batch[:, :-1]
            next_move = generator.generate(input_seq)
            fake_batch = torch.cat([input_seq, next_move.unsqueeze(-1)], dim=1)
            # 更新判別器
            discriminator.train()
            generator.eval()
            opt_d.zero_grad()
            d_loss, _ = gan_loss(discriminator, real_batch, fake_batch, device)
            d_loss.backward()
            opt_d.step()
            d_loss_sum += d_loss.item() * batch_size
            # 更新生成器
            discriminator.eval()
            generator.train()
            opt_g.zero_grad()
            # 重新生成假樣本
            next_move = generator.generate(input_seq)
            fake_batch = torch.cat([input_seq, next_move.unsqueeze(-1)], dim=1)
            _, g_loss = gan_loss(discriminator, real_batch, fake_batch, device)
            g_loss.backward()
            opt_g.step()
            g_loss_sum += g_loss.item() * batch_size
        d_avg = d_loss_sum / len(loader.dataset)
        g_avg = g_loss_sum / len(loader.dataset)
        print(f"[Epoch {epoch+1}/{args.epochs}] D_loss: {d_avg:.4f}, G_loss: {g_avg:.4f}")
    # 儲存模型
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{args.game}_gan.pt")
    torch.save({"generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "vocab": vocab}, model_path)
    print(f"模型已儲存至 {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="訓練 LSTM 模型用於棋類對局預測")
    parser.add_argument("--game", type=str, choices=["chess"], default="chess", help="棋種: chess/go/xiangqi")
    parser.add_argument("--data_dir", type=str, default="../data", help="資料目錄")
    parser.add_argument("--model_dir", type=str, default="../models", help="模型保存目錄")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=40)
    parser.add_argument("--use_gan", action="store_true", help="使用 GAN 對抗訓練")
    args = parser.parse_args()
    if args.use_gan:
        train_with_gan(args)
    else:
        train_generator_only(args)


if __name__ == "__main__":
    main()