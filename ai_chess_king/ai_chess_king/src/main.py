"""
main.py
=======

啟動程式的主入口。支援命令列選項指定棋種、是否啟用對抗學習模型以及網路連線模式。

使用方法：
```
python main.py --game chess
python main.py --game go --server  # 啟動伺服器等待對手連線
python main.py --game xiangqi --client --host 192.168.1.10
```

預設情況下，若未指定 --server 或 --client 則啟動單機模式。模型路徑預設讀取 ``models/{game}_lstm_gan.pth``；若不存在則採用簡易隨機策略。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

try:
    import torch  # type: ignore
except ImportError:
    torch = None  # type: ignore

from gui import GUI
from model import LSTMGenerator
from network import Server, Client
from preprocessing import load_vocab


def load_model(game: str, model_dir: str) -> LSTMGenerator:
    """嘗試載入儲存的模型；若不存在，返回一個隨機初始化模型。"""
    vocab_path = os.path.join(model_dir, f"{game}_vocab.json")
    model_path = os.path.join(model_dir, f"{game}_lstm_gan.pth")
    # 假設需要先載入 vocab 取得大小
    if os.path.exists(vocab_path):
        vocab = load_vocab(vocab_path)
        vocab_size = len(vocab)
    else:
        vocab_size = 500  # fallback 規定字典大小
    model = LSTMGenerator(vocab_size=vocab_size)
    # 如果 pytorch 不可用則直接返回未訓練模型
    if torch is None:
        print("警告：系統未安裝 PyTorch，使用隨機初始化模型")
        return model
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"已載入模型 {model_path}")
    else:
        print(f"警告：找不到模型 {model_path}，使用隨機初始化模型")
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 棋王遊戲入口")
    parser.add_argument("--game", default="chess", choices=["chess", "xiangqi", "go"], help="選擇遊戲種類")
    parser.add_argument("--model_dir", default="models", help="模型儲存目錄")
    parser.add_argument("--server", action="store_true", help="啟動為伺服器模式")
    parser.add_argument("--client", action="store_true", help="啟動為客戶端模式")
    parser.add_argument("--host", default="localhost", help="伺服器主機名稱 (客戶端模式適用)")
    parser.add_argument("--port", type=int, default=5000, help="連線埠號")
    args = parser.parse_args()

    # 載入模型
    model = load_model(args.game, args.model_dir)

    # 建立 GUI
    gui = GUI(game_type=args.game, version=time.strftime("%Y-%m-%d"))

    net_server: Server | None = None
    net_client: Client | None = None

    def handle_network_message(msg: str) -> None:
        """處理收到的棋步訊息：將對手步驟更新到棋盤。"""
        # 預期格式為 piece@(row,col)
        try:
            piece, pos = msg.split('@')
            pos = pos.strip('()')
            row, col = map(int, pos.split(','))
            # 在棋盤放置對方棋子
            gui.board.place_piece(row, col, piece)
            print(f"收到對手步驟：{msg}")
        except Exception as exc:
            print(f"解析網路訊息錯誤：{exc}, 原始訊息: {msg}")

    # 啟動網路模式
    if args.server:
        net_server = Server(host=args.host, port=args.port)
        # 設置回調
        net_server.set_receive_callback(handle_network_message)
        threading.Thread(target=net_server.start, daemon=True).start()
    elif args.client:
        net_client = Client(host=args.host, port=args.port)
        net_client.set_receive_callback(handle_network_message)
        net_client.connect()

    # 內部函式，用於在落子後呼叫 AI 產生回應步驟
    def after_move_callback(row: int, col: int, piece: str) -> None:
        # AI move: 這裡僅示範隨機策略，若欲使用模型可以根據當前棋盤構建序列並呼叫 model.generate
        # 產生下一步：遍歷空格尋找第一個可下的位置
        for r in range(gui.board.rows):
            for c in range(gui.board.cols):
                if gui.board.board[r][c] is None:
                    ai_piece = 'W' if piece == 'B' else 'B'
                    gui.board.place_piece(r, c, ai_piece)
                    # 若有網路連線則傳送
                    message = f"{ai_piece}@({r},{c})"
                    if net_server:
                        net_server.send(message)
                    if net_client:
                        net_client.send(message)
                    return

    # 將 GUI 的點擊處理改造為呼叫 after_move_callback
    original_handle_click = gui.board.handle_click
    def new_handle_click(pos, top_left, board_size):
        cell = original_handle_click(pos, top_left, board_size)
        if cell:
            row, col = cell
            piece = 'B' if len(gui.board.move_log) % 2 == 0 else 'W'
            placed = gui.board.place_piece(row, col, piece)
            if placed:
                # 傳送我的棋步給對方
                message = f"{piece}@({row},{col})"
                if net_server:
                    net_server.send(message)
                if net_client:
                    net_client.send(message)
                # AI 回應
                after_move_callback(row, col, piece)
        return cell
    gui.board.handle_click = new_handle_click  # type: ignore

    # 運行 GUI 主迴圈
    try:
        gui.run()
    finally:
        # 關閉網路
        if net_client:
            net_client.disconnect()
        if net_server:
            net_server.stop()


if __name__ == "__main__":
    main()