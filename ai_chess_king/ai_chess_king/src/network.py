"""
network.py
===========

提供簡易區域網路連線功能，使兩台電腦可以交換棋子訊息。此模組以 TCP socket 實作基本的伺服器和客戶端，並可在遊戲中傳送和接收棋步。

使用方法：

```
from network import Server, Client

# 啟動伺服器
server = Server(host='0.0.0.0', port=5000)
server.start()

# 等待來自客戶端的連線並在遊戲循環中持續使用

```

或者

```
# 連線到伺服器
client = Client(host='192.168.1.10', port=5000)
client.connect()
client.send_move('B@(3,4)')
move = client.receive_move()
```

注意：這個模組僅提供簡單串流文字訊息，沒有加入資料驗證或協議；實際應用時應加上錯誤處理與安全性檢查。
"""

import socket
import threading
from typing import Optional, Callable


class Server:
    """簡易 TCP 伺服器，接受單一連線並收發訊息。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.running = False
        self.receive_callback: Optional[Callable[[str], None]] = None

    def start(self) -> None:
        """啟動伺服器並等待客戶端連線。"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"伺服器啟動，等待連線 ({self.host}:{self.port})...")
        self.client_socket, addr = self.server_socket.accept()
        print(f"客戶端已連線: {addr}")
        self.running = True
        threading.Thread(target=self._receive_thread, daemon=True).start()

    def _receive_thread(self) -> None:
        while self.running and self.client_socket:
            try:
                data = self.client_socket.recv(1024)
                if not data:
                    break
                message = data.decode('utf-8')
                if self.receive_callback:
                    self.receive_callback(message)
            except ConnectionResetError:
                break
        self.running = False
        print("客戶端已離線")

    def send(self, message: str) -> None:
        """發送訊息到客戶端。"""
        if self.client_socket:
            self.client_socket.sendall(message.encode('utf-8'))

    def set_receive_callback(self, callback: Callable[[str], None]) -> None:
        """設定收到訊息時呼叫的回調函式。"""
        self.receive_callback = callback

    def stop(self) -> None:
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()


class Client:
    """簡易 TCP 客戶端，連線至伺服器並收發訊息。"""

    def __init__(self, host: str, port: int = 5000) -> None:
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.receive_callback: Optional[Callable[[str], None]] = None

    def connect(self) -> None:
        """連線至伺服器。"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.running = True
        threading.Thread(target=self._receive_thread, daemon=True).start()
        print(f"已連線至伺服器 ({self.host}:{self.port})")

    def _receive_thread(self) -> None:
        while self.running and self.socket:
            try:
                data = self.socket.recv(1024)
                if not data:
                    break
                message = data.decode('utf-8')
                if self.receive_callback:
                    self.receive_callback(message)
            except ConnectionResetError:
                break
        self.running = False
        print("與伺服器連線中斷")

    def send(self, message: str) -> None:
        """發送訊息到伺服器。"""
        if self.socket:
            self.socket.sendall(message.encode('utf-8'))

    def set_receive_callback(self, callback: Callable[[str], None]) -> None:
        """設定收到訊息時呼叫的回調函式。"""
        self.receive_callback = callback

    def disconnect(self) -> None:
        self.running = False
        if self.socket:
            self.socket.close()