# -*- coding: utf-8 -*-
"""
human VS AI models with pygame GUI
Click on the board to make your move

@author: Junxiao Song
Modified to use pygame interface
"""

from __future__ import print_function
import pickle
import pygame as pg
import numpy as np
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras


class GameMenu(object):
    """
    Game menu for pygame interface
    """
    def __init__(self):
        pg.init()
        self.screen_width = 600
        self.screen_height = 500
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        pg.display.set_caption("AlphaZero Gomoku Game")
        
        # 尝试加载支持中文的字体
        try:
            # Windows系统中文字体
            self.font_large = pg.font.SysFont('simsun', 36)  # 减小字体
            self.font_medium = pg.font.SysFont('simsun', 28)  # 减小字体
            self.font_small = pg.font.SysFont('simsun', 20)
        except:
            try:
                # 备选字体
                self.font_large = pg.font.SysFont('microsoftyaheui', 36)  # 减小字体
                self.font_medium = pg.font.SysFont('microsoftyaheui', 28)  # 减小字体
                self.font_small = pg.font.SysFont('microsoftyaheui', 20)
            except:
                # 最后备选，使用默认字体但调整大小
                self.font_large = pg.font.Font(None, 36)
                self.font_medium = pg.font.Font(None, 28)
                self.font_small = pg.font.Font(None, 20)
        
        self.current_step = "game_type"  # game_type, game_mode, first_player
        self.game_type = None
        self.game_mode = None
        self.human_first = None
        
    def draw_button(self, text, x, y, width, height, color, text_color, selected=False):
        """绘制按钮"""
        if selected:
            pg.draw.rect(self.screen, (100, 150, 255), (x-2, y-2, width+4, height+4))
        pg.draw.rect(self.screen, color, (x, y, width, height))
        pg.draw.rect(self.screen, (0, 0, 0), (x, y, width, height), 2)
        
        text_surface = self.font_medium.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(text_surface, text_rect)
        
        return pg.Rect(x, y, width, height)
    
    def draw_title(self, title):
        """绘制标题"""
        title_surface = self.font_large.render(title, True, (0, 0, 0))
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, 80))
        self.screen.blit(title_surface, title_rect)
    
    def draw_game_type_selection(self):
        """绘制游戏类型选择界面"""
        self.screen.fill((240, 240, 240))
        self.draw_title("Game Type Selection")
        
        # 五子棋按钮
        btn1 = self.draw_button("Gomoku (8x8, 5 in row)", 150, 180, 300, 60, (200, 255, 200), (0, 0, 0))
        
        # 四子棋按钮
        btn2 = self.draw_button("Connect 4 (6x6, 4 in row)", 150, 260, 300, 60, (255, 200, 200), (0, 0, 0))
        
        # 说明文字
        info_text = "Click to select game type"
        info_surface = self.font_small.render(info_text, True, (100, 100, 100))
        info_rect = info_surface.get_rect(center=(self.screen_width // 2, 360))
        self.screen.blit(info_surface, info_rect)
        
        return btn1, btn2
    
    def draw_game_mode_selection(self):
        """绘制游戏模式选择界面"""
        self.screen.fill((240, 240, 240))
        game_name = "Gomoku" if self.game_type == "gomoku" else "Connect 4"
        self.draw_title(f"{game_name} - Game Mode")
        
        # 人类vs AI按钮
        btn1 = self.draw_button("Human vs AI", 150, 180, 300, 60, (200, 255, 200), (0, 0, 0))
        
        # AI自对弈按钮
        btn2 = self.draw_button("AI vs AI", 150, 260, 300, 60, (255, 255, 200), (0, 0, 0))
        
        # 返回按钮
        btn_back = self.draw_button("Back", 50, 400, 100, 40, (220, 220, 220), (0, 0, 0))
        
        return btn1, btn2, btn_back
    
    def draw_first_player_selection(self):
        """绘制先手选择界面"""
        self.screen.fill((240, 240, 240))
        game_name = "Gomoku" if self.game_type == "gomoku" else "Connect 4"
        self.draw_title(f"{game_name} - First Player")
        
        # 人类先手按钮
        btn1 = self.draw_button("Human First (Black)", 150, 180, 300, 60, (100, 100, 100), (255, 255, 255))
        
        # AI先手按钮
        btn2 = self.draw_button("AI First (Black)", 150, 260, 300, 60, (255, 255, 255), (0, 0, 0))
        
        # 返回按钮
        btn_back = self.draw_button("Back", 50, 400, 100, 40, (220, 220, 220), (0, 0, 0))
        
        return btn1, btn2, btn_back
    
    def get_game_settings(self):
        """获取游戏设置"""
        clock = pg.time.Clock()
        
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit()
                elif event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    
                    if self.current_step == "game_type":
                        btn1, btn2 = self.draw_game_type_selection()
                        if btn1.collidepoint(mouse_pos):
                            self.game_type = "gomoku"
                            self.current_step = "game_mode"
                        elif btn2.collidepoint(mouse_pos):
                            self.game_type = "connect4"
                            self.current_step = "game_mode"
                            
                    elif self.current_step == "game_mode":
                        btn1, btn2, btn_back = self.draw_game_mode_selection()
                        if btn1.collidepoint(mouse_pos):
                            self.game_mode = "human_vs_ai"
                            self.current_step = "first_player"
                        elif btn2.collidepoint(mouse_pos):
                            self.game_mode = "ai_vs_ai"
                            self.human_first = None
                            return self.get_settings_result()
                        elif btn_back.collidepoint(mouse_pos):
                            self.current_step = "game_type"
                            
                    elif self.current_step == "first_player":
                        btn1, btn2, btn_back = self.draw_first_player_selection()
                        if btn1.collidepoint(mouse_pos):
                            self.human_first = True
                            return self.get_settings_result()
                        elif btn2.collidepoint(mouse_pos):
                            self.human_first = False
                            return self.get_settings_result()
                        elif btn_back.collidepoint(mouse_pos):
                            self.current_step = "game_mode"
            
            # 绘制当前界面
            if self.current_step == "game_type":
                self.draw_game_type_selection()
            elif self.current_step == "game_mode":
                self.draw_game_mode_selection()
            elif self.current_step == "first_player":
                self.draw_first_player_selection()
                
            pg.display.update()
            clock.tick(60)
    
    def get_settings_result(self):
        """返回设置结果"""
        if self.game_type == "gomoku":
            n_in_row = 5
            width, height = 8, 8
            model_file = 'best_policy_8_8_5.model'
            game_type_name = "Gomoku"
        else:  # connect4
            n_in_row = 4
            width, height = 6, 6
            model_file = 'best_policy_6_6_4.model'
            game_type_name = "Connect 4"
        
        pg.quit()  # 关闭菜单窗口
        return n_in_row, width, height, model_file, self.game_mode, self.human_first, game_type_name


class Game_UI(object):
    """
    Game UI for pygame visualization
    """
    def __init__(self, board, is_shown=1):
        self.board = board
        self.is_shown = is_shown
        self.width = board.width
        self.height = board.height
        
        if self.is_shown:
            pg.init()
            self.screen_width = (self.width + 1) * 40
            self.screen_height = (self.height + 1) * 40 + 60  # Extra space for status
            self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
            pg.display.set_caption("AlphaZero Gomoku")
            
            # 尝试加载支持中文的字体
            try:
                self.font = pg.font.SysFont('simsun', 24)  # 减小字体
            except:
                try:
                    self.font = pg.font.SysFont('microsoftyaheui', 24)  # 减小字体
                except:
                    self.font = pg.font.Font(None, 24)
            
    def draw(self):
        """Draw the board and pieces"""
        if not self.is_shown:
            return
            
        # Clear screen
        self.screen.fill((200, 200, 200))
        
        # Draw board lines
        for i in range(self.width):
            pg.draw.line(self.screen, (0, 0, 0), 
                        (40 + i * 40, 40), 
                        (40 + i * 40, 40 + (self.height - 1) * 40), 2)
        for i in range(self.height):
            pg.draw.line(self.screen, (0, 0, 0), 
                        (40, 40 + i * 40), 
                        (40 + (self.width - 1) * 40, 40 + i * 40), 2)
        
        # Draw pieces
        for move, player in self.board.states.items():
            h, w = self.board.move_to_location(move)
            x = 40 + w * 40
            y = 40 + h * 40
            if player == 1:
                pg.draw.circle(self.screen, (0, 0, 0), (x, y), 15)  # Black piece
            else:
                pg.draw.circle(self.screen, (255, 255, 255), (x, y), 15)  # White piece
                pg.draw.circle(self.screen, (0, 0, 0), (x, y), 15, 2)  # Border
                
        # Draw current player info
        current_player_text = "Current Player: " + ("Black" if self.board.current_player == 1 else "White")
        text_surface = self.font.render(current_player_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, self.screen_height - 50))
        
    def draw_result(self, winner):
        """Draw game result with restart option"""
        if not self.is_shown:
            return
            
        if winner == -1:
            result_text = "Draw!"
        elif winner == 1:
            result_text = "Black Wins!"
        else:
            result_text = "White Wins!"
            
        # 使用更大的字体显示结果
        try:
            big_font = pg.font.SysFont('simsun', 36)  # 减小字体
        except:
            try:
                big_font = pg.font.SysFont('microsoftyaheui', 36)  # 减小字体
            except:
                big_font = pg.font.Font(None, 36)
        
        text_surface = big_font.render(result_text, True, (255, 0, 0))
        text_rect = text_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 40))
        
        # Draw semi-transparent overlay
        overlay = pg.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(180)
        overlay.fill((255, 255, 255))
        self.screen.blit(overlay, (0, 0))
        
        # Draw result text with background
        bg_rect = text_rect.inflate(40, 20)
        pg.draw.rect(self.screen, (255, 255, 255), bg_rect)
        pg.draw.rect(self.screen, (0, 0, 0), bg_rect, 3)
        self.screen.blit(text_surface, text_rect)
        
        # Draw restart button
        restart_btn = pg.Rect(self.screen_width // 2 - 100, text_rect.bottom + 30, 200, 40)
        pg.draw.rect(self.screen, (100, 255, 100), restart_btn)
        pg.draw.rect(self.screen, (0, 0, 0), restart_btn, 2)
        restart_text = self.font.render("New Game", True, (0, 0, 0))
        restart_text_rect = restart_text.get_rect(center=restart_btn.center)
        self.screen.blit(restart_text, restart_text_rect)
        
        # Draw exit button
        exit_btn = pg.Rect(self.screen_width // 2 - 50, restart_btn.bottom + 10, 100, 30)
        pg.draw.rect(self.screen, (255, 100, 100), exit_btn)
        pg.draw.rect(self.screen, (0, 0, 0), exit_btn, 2)
        exit_text = self.font.render("Exit", True, (0, 0, 0))
        exit_text_rect = exit_text.get_rect(center=exit_btn.center)
        self.screen.blit(exit_text, exit_text_rect)
        
        return restart_btn, exit_btn


class Human(object):
    """
    human player with pygame interface
    """

    def __init__(self):
        self.player = None
        self.move = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        """Get action from mouse click"""
        self.move = None
        
        while self.move is None:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit()
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    x, y = event.pos
                    # Convert pixel coordinates to board coordinates
                    board_x = round((x - 40) / 40)
                    board_y = round((y - 40) / 40)
                    
                    # Check if click is within board bounds
                    if 0 <= board_x < board.width and 0 <= board_y < board.height:
                        move = board_y * board.width + board_x
                        if move in board.availables:
                            self.move = move
                            return move
                        else:
                            print("Position already occupied!")
                    else:
                        print("Click within the board!")
                        
        return self.move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    while True:  # 主游戏循环
        # 使用pygame界面获取游戏设置
        menu = GameMenu()
        n_in_row, width, height, model_file, game_mode, human_first, game_type = menu.get_game_settings()
        
        print(f"\n正在启动 {game_type} 游戏...")
        print(f"棋盘大小: {width}x{height}")
        print(f"获胜条件: {n_in_row}子连珠")
        print(f"使用模型: {model_file}")
        print(f"游戏模式: {'人类 vs AI' if game_mode == 'human_vs_ai' else 'AI 自对弈'}")
        
        try:
            board = Board(width=width, height=height, n_in_row=n_in_row)
            
            # Initialize pygame UI
            game_ui = Game_UI(board, is_shown=1)

            # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
            try:
                policy_param = pickle.load(open(model_file, 'rb'))
            except:
                policy_param = pickle.load(open(model_file, 'rb'),
                                           encoding='bytes')  # To support python3
            best_policy = PolicyValueNetNumpy(width, height, policy_param)
            mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                     c_puct=5,
                                     n_playout=400)  # set larger n_playout for better performance

            restart_game = True
            while restart_game:
                if game_mode == "human_vs_ai":
                    # 人类 vs AI 模式
                    human = Human()
                    
                    # 根据先手设置玩家ID
                    if human_first:
                        human.set_player_ind(1)  # 人类是黑子(1)
                        mcts_player.set_player_ind(2)  # AI是白子(2)
                        start_player = 0  # 玩家1先手
                    else:
                        human.set_player_ind(2)  # 人类是白子(2)
                        mcts_player.set_player_ind(1)  # AI是黑子(1)
                        start_player = 0  # 玩家1(AI)先手

                    board.init_board(start_player=start_player)
                    
                    # Draw initial board
                    game_ui.draw()
                    pg.display.update()
                    
                    if human_first:
                        print("游戏开始! 你是黑子先手，点击棋盘落子。")
                    else:
                        print("游戏开始! AI是黑子先手，你是白子。")
                    
                    # 人类 vs AI 游戏循环
                    restart_game = run_human_vs_ai(board, game_ui, human, mcts_player)
                    
                else:
                    # AI vs AI 模式
                    ai_player1 = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
                    ai_player2 = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
                    ai_player1.set_player_ind(1)
                    ai_player2.set_player_ind(2)
                    
                    board.init_board(start_player=0)
                    game_ui.draw()
                    pg.display.update()
                    
                    print("AI 自对弈开始! AI1(黑子) vs AI2(白子)")
                    
                    # AI vs AI 游戏循环
                    restart_game = run_ai_vs_ai(board, game_ui, ai_player1, ai_player2)
                
                # 如果选择重新开始，重置棋盘
                if restart_game:
                    board.init_board(start_player=start_player if game_mode == "human_vs_ai" else 0)
                    
        except KeyboardInterrupt:
            print('\n\rquit')
            break
        finally:
            pg.quit()
            
        # 如果没有选择重新开始，退出主循环
        if not restart_game:
            break


def run_human_vs_ai(board, game_ui, human, mcts_player):
    """人类 vs AI 游戏循环"""
    while True:
        current_player = board.current_player
        
        if current_player == human.player:  # 人类回合
            print("轮到你了...")
            move = human.get_action(board)
            board.do_move(move)
            game_ui.draw()
            pg.display.update()
            
            # Check if game ends
            end, winner = board.game_end()
            if end:
                restart = handle_game_end(game_ui, winner, "human_vs_ai", human.player)
                return restart  # 返回是否重新开始
                
        else:  # AI 回合
            print("AI 思考中...")
            move = mcts_player.get_action(board)
            board.do_move(move)
            print(f"AI 落子位置: {board.move_to_location(move)}")
            game_ui.draw()
            pg.display.update()
            
            # Check if game ends
            end, winner = board.game_end()
            if end:
                restart = handle_game_end(game_ui, winner, "human_vs_ai", human.player)
                return restart  # 返回是否重新开始


def run_ai_vs_ai(board, game_ui, ai_player1, ai_player2):
    """AI vs AI 游戏循环"""
    move_count = 0
    while True:
        current_player = board.current_player
        
        if current_player == 1:  # AI1 回合
            print("AI1 (黑子) 思考中...")
            move = ai_player1.get_action(board)
            player_name = "AI1"
        else:  # AI2 回合
            print("AI2 (白子) 思考中...")
            move = ai_player2.get_action(board)
            player_name = "AI2"
            
        board.do_move(move)
        move_count += 1
        print(f"{player_name} 第{move_count}手落子位置: {board.move_to_location(move)}")
        game_ui.draw()
        pg.display.update()
        
        # 在AI对弈模式下添加延迟，便于观看
        pg.time.wait(1000)  # 等待1秒
        
        # 处理pygame事件（允许用户关闭窗口）
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
        
        # Check if game ends
        end, winner = board.game_end()
        if end:
            restart = handle_game_end(game_ui, winner, "ai_vs_ai", 1)
            return restart  # 返回是否重新开始


def handle_game_end(game_ui, winner, mode, human_player_id=1):
    """处理游戏结束"""
    if winner != -1:
        if mode == "human_vs_ai":
            if winner == human_player_id:
                print("游戏结束! 人类获胜!")
            else:
                print("游戏结束! AI获胜!")
        else:  # ai_vs_ai
            print(f"游戏结束! AI{winner} 获胜!")
    else:
        print("游戏结束! 平局!")
    
    restart_btn, exit_btn = game_ui.draw_result(winner)
    pg.display.update()
    
    # Wait for user choice
    waiting = True
    while waiting:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                waiting = False
                return False  # 退出游戏
            elif event.type == pg.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if restart_btn.collidepoint(mouse_pos):
                    print("重新开始游戏...")
                    waiting = False
                    return True  # 重新开始游戏
                elif exit_btn.collidepoint(mouse_pos):
                    waiting = False
                    return False  # 退出游戏
    
    return False  # 默认退出


if __name__ == '__main__':
    run()
