"""
gui.py
======

使用 ``pygame`` 實作簡易圖形介面，支援三種棋類（西洋棋、象棋、圍棋）的棋盤繪製、基礎操作以及選單功能。

介面分為左右兩個面板：

* 左側面板顯示遊戲標題、選項選單（遊戲、連線、狀態、版本）。
* 右側面板包含名稱輸入框、計時資訊、落子記錄以及操作按鈕。

由於完整的 AI 對弈邏輯與網路連線在 ``main.py`` 中整合，這裡專注於視覺呈現與使用者互動。

使用本模組前請先安裝 ``pygame``：
```
pip install pygame
```

注：如果您在無視覺環境中執行本程式，Pygame 可能無法正常初始，請確保已在圖形桌面環境或支援 SDL 的系統上運行。
"""

from __future__ import annotations

import pygame
import random
import string
from typing import List, Tuple, Optional, Callable


class Button:
    """簡易按鈕類別。

    Args:
        rect: ``pygame.Rect`` 定義位置與尺寸。
        text: 按鈕文字。
        callback: 點擊時呼叫的函式。
    """

    def __init__(self, rect: pygame.Rect, text: str, callback: Callable[[], None], font: pygame.font.Font,
                 color: Tuple[int, int, int] = (200, 200, 200), text_color: Tuple[int, int, int] = (0, 0, 0)):
        self.rect = rect
        self.text = text
        self.callback = callback
        self.font = font
        self.color = color
        self.text_color = text_color

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, self.color, self.rect)
        label = self.font.render(self.text, True, self.text_color)
        label_rect = label.get_rect(center=self.rect.center)
        surface.blit(label, label_rect)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()


class TextInput:
    """簡單的文字輸入框，用於輸入玩家名稱。"""

    def __init__(self, rect: pygame.Rect, font: pygame.font.Font, initial_text: str = ""):
        self.rect = rect
        self.font = font
        self.text = initial_text
        self.active = False
        self.bg_color_inactive = (240, 240, 240)
        self.bg_color_active = (220, 220, 255)
        self.text_color = (0, 0, 0)

    def draw(self, surface: pygame.Surface) -> None:
        color = self.bg_color_active if self.active else self.bg_color_inactive
        pygame.draw.rect(surface, color, self.rect)
        # 文字略為留白
        pad = 5
        text_surf = self.font.render(self.text, True, self.text_color)
        surface.blit(text_surf, (self.rect.x + pad, self.rect.y + (self.rect.height - text_surf.get_height()) // 2))

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                if len(self.text) < 20:
                    self.text += event.unicode

    def get_text(self) -> str:
        return self.text


class GameBoard:
    """支援三種棋盤的通用棋盤類別。"""

    def __init__(self, game_type: str = "chess") -> None:
        # game_type: 'chess' (西洋棋), 'xiangqi' (象棋), 'go' (圍棋)
        self.game_type = game_type
        self.reset()

    def reset(self) -> None:
        if self.game_type == "chess":
            self.rows, self.cols = 8, 8
        elif self.game_type == "xiangqi":
            self.rows, self.cols = 10, 9
        elif self.game_type == "go":
            self.rows, self.cols = 19, 19
        else:
            raise ValueError(f"Unsupported game type: {self.game_type}")
        # 初始化棋盤狀態為 None
        self.board = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.move_log: List[str] = []  # 簡易落子記錄

    def handle_click(self, pos: Tuple[int, int], top_left: Tuple[int, int], board_size: int) -> Optional[Tuple[int, int]]:
        """根據滑鼠座標計算棋盤格子座標。

        Args:
            pos: 滑鼠位置
            top_left: 棋盤左上角座標
            board_size: 棋盤區域的寬度（高度）

        Returns:
            (row, col) if 點擊在棋盤內，否則 None
        """
        x, y = pos
        bx, by = top_left
        cell_size = board_size / max(self.rows, self.cols)
        if bx <= x < bx + cell_size * self.cols and by <= y < by + cell_size * self.rows:
            col = int((x - bx) // cell_size)
            row = int((y - by) // cell_size)
            return row, col
        return None

    def place_piece(self, row: int, col: int, piece: str) -> bool:
        """在棋盤上放置棋子，成功返回 True，若該位置已有棋子返回 False。"""
        if self.board[row][col] is None:
            self.board[row][col] = piece
            # 加入簡易記錄：行列+棋子
            self.move_log.append(f"{piece}@({row},{col})")
            return True
        return False

    def draw(self, surface: pygame.Surface, top_left: Tuple[int, int], board_size: int) -> None:
        """繪製棋盤與棋子。"""
        bx, by = top_left
        max_dim = max(self.rows, self.cols)
        cell_size = board_size / max_dim
        # 繪製格線
        for r in range(self.rows + 1):
            start_pos = (bx, by + r * cell_size)
            end_pos = (bx + self.cols * cell_size, by + r * cell_size)
            pygame.draw.line(surface, (0, 0, 0), start_pos, end_pos, 1)
        for c in range(self.cols + 1):
            start_pos = (bx + c * cell_size, by)
            end_pos = (bx + c * cell_size, by + self.rows * cell_size)
            pygame.draw.line(surface, (0, 0, 0), start_pos, end_pos, 1)
        # 繪製棋子（簡易標示）
        for r in range(self.rows):
            for c in range(self.cols):
                piece = self.board[r][c]
                if piece is not None:
                    center_x = bx + c * cell_size + cell_size / 2
                    center_y = by + r * cell_size + cell_size / 2
                    radius = cell_size * 0.35
                    if piece == 'B':
                        color = (0, 0, 0)
                    elif piece == 'W':
                        color = (255, 255, 255)
                    else:
                        # 其它棋種使用不同顏色
                        color = (200, 100, 50)
                    pygame.draw.circle(surface, color, (int(center_x), int(center_y)), int(radius))


def random_name() -> str:
    """生成一個由兩個字母和 9 個數字組成的隨機名稱。"""
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    numbers = ''.join(random.choices(string.digits, k=9))
    return letters + numbers


class GUI:
    """整合選單和棋盤的主介面。

    Args:
        game_type: 遊戲種類：``"chess"``、``"xiangqi"``或``"go"``。
        version: 顯示的版本號。
        on_user_move: 當使用者在棋盤上成功放置棋子後回呼函式。函式接受三個參數：行(row)、列(col)、棋子名稱(piece)。
    """

    def __init__(self, game_type: str = "chess", version: str = "v1.0", on_user_move: Optional[Callable[[int, int, str], None]] = None):
        pygame.init()
        # Window dimensions
        self.width = 1000
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Let's Go")

        # 字體
        self.font_title = pygame.font.SysFont(None, 48)
        self.font_subtitle = pygame.font.SysFont(None, 28)
        self.font_small = pygame.font.SysFont(None, 20)

        # game board
        self.board = GameBoard(game_type)
        self.game_type = game_type

        self.version = version
        self.on_user_move = on_user_move

        # 名稱輸入框
        self.name_input = TextInput(pygame.Rect(730, 50, 250, 40), self.font_subtitle)
        # 計時顯示
        self.show_timer = True
        self.elapsed_time_p1 = 0.0
        self.elapsed_time_p2 = 0.0
        # 落子記錄從棋盤獲取

        # 左側選單按鈕
        self.buttons: List[Button] = []
        self.setup_buttons()

        self.clock = pygame.time.Clock()

        self.running = True

    def setup_buttons(self) -> None:
        # 生成各種功能按鈕
        btn_width = 90
        btn_height = 30
        # 右下功能按鈕
        y_base = 600
        gap = 10
        x_start = 720

        def resign():
            print("玩家投降")

        def pass_move():
            print("pass/和棋")

        def pause():
            print("暫停")

        def new_game():
            self.board.reset()
            print("開始新局")

        # 建立按鈕實例
        self.buttons.append(Button(pygame.Rect(x_start, y_base, btn_width, btn_height), "投降", resign, self.font_small))
        self.buttons.append(Button(pygame.Rect(x_start + (btn_width + gap), y_base, btn_width, btn_height), "PASS", pass_move, self.font_small))
        self.buttons.append(Button(pygame.Rect(x_start + 2 * (btn_width + gap), y_base, btn_width, btn_height), "暫停", pause, self.font_small))
        self.buttons.append(Button(pygame.Rect(x_start + 3 * (btn_width + gap), y_base, btn_width, btn_height), "新局", new_game, self.font_small))

    def run(self) -> None:
        """開始主迴圈。"""
        start_ticks = pygame.time.get_ticks()
        while self.running:
            delta_time = self.clock.tick(30) / 1000.0  # 每幀秒數
            # 更新時間
            self.elapsed_time_p1 += delta_time
            self.elapsed_time_p2 += delta_time
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # 處理按鈕事件
                for btn in self.buttons:
                    btn.handle_event(event)
                # 名稱輸入框
                self.name_input.handle_event(event)
                # 棋盤點擊
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    board_top_left = (50, 150)
                    board_size = 500
                    cell = self.board.handle_click(event.pos, board_top_left, board_size)
                    if cell:
                        row, col = cell
                        # 依照棋種簡單交替放置黑白棋；此處僅作示例
                        # 奇數步放黑棋 B，偶數步放白棋 W
                        piece = 'B' if len(self.board.move_log) % 2 == 0 else 'W'
                        placed = self.board.place_piece(row, col, piece)
                        if placed:
                            print(f"放置 {piece} 於 {row},{col}")
                            # 回呼函式
                            if self.on_user_move:
                                self.on_user_move(row, col, piece)

            # 填滿背景
            self.screen.fill((250, 250, 250))
            # 左側面板
            self.draw_left_panel()
            # 棋盤
            self.board.draw(self.screen, (50, 150), 500)
            # 右側面板
            self.draw_right_panel()
            # 更新顯示
            pygame.display.flip()
        pygame.quit()

    def draw_left_panel(self) -> None:
        # 遊戲名稱
        title_surf = self.font_title.render("Let's Go", True, (0, 0, 0))
        self.screen.blit(title_surf, (50, 20))
        # 選單
        y = 70
        # 遊戲選項
        self.screen.blit(self.font_subtitle.render("1. 遊戲", True, (0, 0, 0)), (50, y))
        y += 25
        for idx, name in enumerate(["1.1 西洋棋", "1.2 象棋", "1.3 圍棋"]):
            color = (0, 0, 255) if (self.game_type == "chess" and idx == 0) or (self.game_type == "xiangqi" and idx == 1) or (self.game_type == "go" and idx == 2) else (0, 0, 0)
            self.screen.blit(self.font_small.render(name, True, color), (70, y))
            y += 20
        y += 10
        # 連線選項
        self.screen.blit(self.font_subtitle.render("2. 連線", True, (0, 0, 0)), (50, y))
        y += 25
        for name in ["2.1 區域網路", "2.2 指定區域", "2.3 公開伺服器"]:
            self.screen.blit(self.font_small.render(name, True, (0, 0, 0)), (70, y))
            y += 20
        y += 10
        # 狀態
        self.screen.blit(self.font_subtitle.render("3. 狀態", True, (0, 0, 0)), (50, y))
        y += 25
        status = "單機"  # 可在主程式依需要更新
        self.screen.blit(self.font_small.render(status, True, (0, 0, 0)), (70, y))
        y += 30
        # 版本
        self.screen.blit(self.font_subtitle.render("4. 版本號", True, (0, 0, 0)), (50, y))
        y += 25
        self.screen.blit(self.font_small.render(self.version, True, (0, 0, 0)), (70, y))

    def draw_right_panel(self) -> None:
        # 名稱輸入
        title = self.font_subtitle.render("輸入名稱：", True, (0, 0, 0))
        self.screen.blit(title, (730, 20))
        self.name_input.draw(self.screen)
        # 如果沒有輸入名稱則顯示自動生成
        name = self.name_input.get_text().strip() or random_name()
        self.screen.blit(self.font_small.render(f"當前名稱: {name}", True, (100, 100, 100)), (730, 100))
        # 計時顯示
        if self.show_timer:
            p1_time = int(self.elapsed_time_p1)
            p2_time = int(self.elapsed_time_p2)
            timer_surf = self.font_small.render(f"P1 時間: {p1_time}s    P2 時間: {p2_time}s", True, (0, 0, 0))
            self.screen.blit(timer_surf, (730, 140))
        # 落子記錄
        self.screen.blit(self.font_subtitle.render("落子紀錄：", True, (0, 0, 0)), (730, 180))
        log_y = 210
        # 顯示最近 10 行
        for move in self.board.move_log[-12:]:
            move_surf = self.font_small.render(move, True, (0, 0, 0))
            self.screen.blit(move_surf, (730, log_y))
            log_y += 18
        # 按鈕繪製
        for btn in self.buttons:
            btn.draw(self.screen)


if __name__ == "__main__":
    # 測試 GUI
    gui = GUI(game_type="go", version="2025-08-17")
    gui.run()