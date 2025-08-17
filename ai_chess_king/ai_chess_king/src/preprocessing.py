"""
preprocessing.py
=================

此模組負責將不同棋類的棋譜讀取並轉換為序列化的整數編碼，方便 LSTM 模型進行訓練。

支援的棋類：

* go    ：讀取 SGF 檔案，透過外部工具轉換為 GTP 格式後，再映射為整數序列。
* chess ：讀取 PGN 檔案，解析為 UCI 代碼序列。
* xiangqi：讀取 PGN 或自定義格式的中國象棋棋譜，解析為對應的整數序列。

為簡化專案示例，此處僅示範如何建立 move→index 的對映表，實際的解析需要使用專門的棋譜解析庫（如 `python-chess` 或自定義解析器）。
"""
from __future__ import annotations

import os
import json
from typing import List, Dict, Tuple, Iterable
import numpy as np

try:
    import chess.pgn  # type: ignore
except ImportError:
    chess = None  # 避免導入失敗


def build_vocab(moves: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """根據所有出現過的棋招建立字典。

    參數
    ------
    moves: 迭代器，包含資料集中所有的招法字串。

    回傳
    ------
    move2idx: 字串到整數編碼的對映。
    idx2move: 整數編碼到字串的反向對映。
    """
    unique = sorted(set(moves))
    move2idx = {m: i + 1 for i, m in enumerate(unique)}  # 0 保留給 PAD
    idx2move = {i: m for m, i in move2idx.items()}
    return move2idx, idx2move


def encode_moves(moves: List[str], move2idx: Dict[str, int], max_length: int) -> np.ndarray:
    """將招法字串列表編碼為固定長度的整數序列。

    不足長度以 0（PAD）補齊，超過長度則截斷。
    """
    seq = [move2idx.get(m, 0) for m in moves[:max_length]]
    if len(seq) < max_length:
        seq.extend([0] * (max_length - len(seq)))
    return np.array(seq, dtype=np.int32)


def decode_moves(indices: List[int], idx2move: Dict[int, str]) -> List[str]:
    """將整數序列轉換回棋招字串。忽略 0 (PAD)。"""
    return [idx2move.get(i, "") for i in indices if i != 0]


def save_vocab(move2idx: Dict[str, int], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(move2idx, f, ensure_ascii=False, indent=2)


def load_vocab(filepath: str) -> Dict[str, int]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def process_chess_pgn(pgn_path: str, max_games: int = 1000, max_length: int = 40) -> Tuple[np.ndarray, Dict[str, int]]:
    """範例：讀取 PGN 檔案中的多盤國際象棋對局，建立詞彙並編碼。

    此函式僅做示範，每盤對局只擷取前 `max_length` 手。
    若安裝了 `python-chess` 套件，會自動解析 PGN；否則將傳回空陣列與空字典。
    """
    if chess is None:
        print("python-chess 未安裝，無法解析 PGN。")
        return np.zeros((0, max_length), dtype=np.int32), {}

    all_moves: List[List[str]] = []
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(max_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            moves = []
            node = game
            while not node.is_end():
                next_node = node.variation(0)
                move = next_node.move.uci()
                moves.append(move)
                node = next_node
            all_moves.append(moves)

    # 建立字典
    vocab_moves = [m for moves in all_moves for m in moves]
    move2idx, _ = build_vocab(vocab_moves)

    # 編碼
    encoded = np.zeros((len(all_moves), max_length), dtype=np.int32)
    for i, mv in enumerate(all_moves):
        encoded[i] = encode_moves(mv, move2idx, max_length)

    return encoded, move2idx


if __name__ == "__main__":
    # 簡易測試：若有 sample.pgn 可跑此程式
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=str, default="sample.pgn", help="PGN 檔案路徑")
    parser.add_argument("--max_games", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=40)
    args = parser.parse_args()
    data, vocab = process_chess_pgn(args.pgn, args.max_games, args.max_length)
    print(f"讀取 {data.shape[0]} 局，每局長度 {data.shape[1]}")
    print(f"字典大小: {len(vocab)}")