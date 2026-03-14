"""
hvp2_game.py — P1 vs. P2 dual-field Snake game
===================================================
Window: 1000 × 540 px  (500 left | thin divider | 500 right)
        + 40 px header bar

Constants
---------
FIELD_W = 500  (25 × 20px cells per side)
FIELD_H = 500  (25 × 20px cells)
DIV_X   = 500  (divider boundary — lethal for both on contact)

Divider wall
------------
Visual : 4 px blue line at x = 500
Hitbox : P1 dies at x >= 500 | P2 dies at x < 500
         (snake enters the cell contp2ning the divider → game over)

P2 virtual coordinates
----------------------
Model sees x in [0, 500) — same as its trp2ning space.
Real P2 x is in [500, 1000).  offset = 500.
  virtual_x = real_x - 500
"""

import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
import json
import os

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
BLOCK   = 20
HEADER  = 40
FIELD_W = 500          # 25 cells wide per side
FIELD_H = 500          # 25 cells tall (play area)
DIV_X   = FIELD_W      # = 500  (divider / boundary)
DIV_VIS = 4            # visual divider width in pixels
WIN_W   = FIELD_W * 2  # = 1000
WIN_H   = HEADER + FIELD_H  # = 540

P2_OFFSET = DIV_X      # subtract from real P2 x → virtual x for model


# ──────────────────────────────────────────────
# Colours
# ──────────────────────────────────────────────
class C:
    BG         = (15,  15,  25)
    GRID       = (22,  22,  38)
    HEADER_BG  = (10,  10,  20)
    DIVIDER    = (70,  100, 240)
    WALL       = (70,   70, 160)

    H_HEAD  = (80,  220,  80)
    H_BODY1 = (40,  160,  40)
    H_BODY2 = (25,  110,  25)

    P2_HEAD  = (80,  180, 255)
    P2_BODY1 = (40,  110, 200)
    P2_BODY2 = (25,   70, 150)

    FOOD_H  = (220,  55,  55)
    FOOD_P2 = (255, 165,   0)

    TEXT     = (200, 200, 200)
    DEAD_DIM = (0,   0,   0,  140)   # SRCALPHA overlay for dead side
    H_SCORE  = (120, 230, 120)
    P2_SCORE = (100, 180, 255)
    BORDER   = (50,  50,  80)
    
    BTN_BG   = (40, 40, 60)
    BTN_HOV  = (60, 60, 85)
    BTN_TXT  = (220, 220, 220)
    
SCORES_FILE = "scores_hvh.json"



# ──────────────────────────────────────────────
# Shared types
# ──────────────────────────────────────────────
class Direction(Enum):
    RIGHT = 1;  LEFT = 2;  UP = 3;  DOWN = 4

Point = namedtuple('Point', ['x', 'y'])

_CLOCK = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


def _resolve_dir(current: Direction, action: list[int]) -> Direction:
    idx = _CLOCK.index(current)
    if   action[1]: idx = (idx + 1) % 4
    elif action[2]: idx = (idx - 1) % 4
    return _CLOCK[idx]


def _move_pt(head: Point, direction: Direction) -> Point:
    x, y = head
    if   direction == Direction.RIGHT: x += BLOCK
    elif direction == Direction.LEFT:  x -= BLOCK
    elif direction == Direction.UP:    y -= BLOCK
    elif direction == Direction.DOWN:  y += BLOCK
    return Point(x, y)


# ──────────────────────────────────────────────
# HumanVsHumanGame
# ──────────────────────────────────────────────
class HumanVsHumanGame:
    """
    Dual-field Snake.

    Each side plays independently.  When one snake dies its side dims and
    shows "ELIMINATED" but the other continues.  Game ends only when both
    are dead, then show_result_screen() is called.
    """

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake — P1 vs. P2")
        self.display = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock   = pygame.time.Clock()

        self.font_score = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_label = pygame.font.SysFont("monospace", 12)
        self.font_big   = pygame.font.SysFont("monospace", 40, bold=True)
        self.font_med   = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_btn   = pygame.font.SysFont("monospace", 17, bold=True)
        self.font_elim  = pygame.font.SysFont("monospace", 22, bold=True)

        self._dead_surf_l = pygame.Surface((FIELD_W, FIELD_H), pygame.SRCALPHA)
        self._dead_surf_l.fill(C.DEAD_DIM)
        self._dead_surf_r = pygame.Surface((FIELD_W, FIELD_H), pygame.SRCALPHA)
        self._dead_surf_r.fill(C.DEAD_DIM)
        
        self.waiting_to_start = False
        self._load_scores()

        self.reset()

    # ─────────────────────────────────────────
    # Scores (Leaderboard)
    # ─────────────────────────────────────────

    def _load_scores(self) -> None:
        if os.path.exists(SCORES_FILE):
            try:
                with open(SCORES_FILE, 'r') as f:
                    data = json.load(f)
                    p1_scores = data.get("human", [{"score": 0, "name": "Anonymous"}] * 10)
                    p2_scores = data.get("human2", [{"score": 0, "name": "Anonymous"}] * 10)
                    
                    if p1_scores and isinstance(p1_scores[0], int):
                        p1_scores = [{"score": s, "name": "Anonymous"} for s in p1_scores]
                    if p2_scores and isinstance(p2_scores[0], int):
                        p2_scores = [{"score": s, "name": "Anonymous"} for s in p2_scores]
                    
                    p1_scores = sorted(p1_scores, key=lambda x: x["score"], reverse=True)[:10]
                    p2_scores = sorted(p2_scores, key=lambda x: x["score"], reverse=True)[:10]
                    
                    self.high_scores = {
                        "p1": p1_scores,
                        "p2": p2_scores
                    }
            except Exception:
                self.high_scores = {"p1": [{"score": 0, "name": "Anonymous"}] * 10, "p2": [{"score": 0, "name": "Anonymous"}] * 10}
        else:
            self.high_scores = {"p1": [{"score": 0, "name": "Anonymous"}] * 10, "p2": [{"score": 0, "name": "Anonymous"}] * 10}
            self._save_scores()

    def _save_scores(self) -> None:
        try:
            with open(SCORES_FILE, 'r+') as f:
                data = json.load(f)
                data["human"] = self.high_scores["p1"]
                data["human2"] = self.high_scores["p2"]
                f.seek(0)
                json.dump(data, f)
                f.truncate()
        except FileNotFoundError:
            try:
                with open(SCORES_FILE, 'w') as f:
                    data = {"human": self.high_scores["p1"], "human2": self.high_scores["p2"]}
                    json.dump(data, f)
            except Exception as e:
                print(f"Error saving scores: {e}")
        except Exception as e:
            print(f"Error saving scores: {e}")

    def _add_scores(self) -> None:
        # Handled in UI
        pass

    # ─────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────

    def reset(self) -> None:
        self.frame = 0

        # P1 snake — centre of left field
        hx = (FIELD_W // 2 // BLOCK) * BLOCK           # 240
        hy = (FIELD_H // 2 // BLOCK) * BLOCK + HEADER  # 260
        self.p1_dir   = Direction.RIGHT
        self.p1_snake = [Point(hx, hy), Point(hx-BLOCK, hy), Point(hx-2*BLOCK, hy)]
        self.p1_head  = self.p1_snake[0]
        self.p1_score = 0
        self.p1_alive = True

        # P2 snake — centre of right field
        ax = DIV_X + (FIELD_W // 2 // BLOCK) * BLOCK   # 740
        self.p2_dir   = Direction.RIGHT
        self.p2_snake = [Point(ax, hy), Point(ax-BLOCK, hy), Point(ax-2*BLOCK, hy)]
        self.p2_head  = self.p2_snake[0]
        self.p2_score = 0
        self.p2_alive = True
        
        self.waiting_to_start = True

        self._place_food_p1()
        self._place_food_p2()

    # ─────────────────────────────────────────
    # Food
    # ─────────────────────────────────────────

    def _place_food_p1(self) -> None:
        while True:
            pt = Point(random.randrange(BLOCK, FIELD_W - BLOCK, BLOCK),
                       random.randrange(HEADER + BLOCK, WIN_H - BLOCK, BLOCK))
            if pt not in self.p1_snake:
                self.p1_food = pt; break

    def _place_food_p2(self) -> None:
        while True:
            pt = Point(random.randrange(DIV_X + BLOCK, WIN_W - BLOCK, BLOCK),
                       random.randrange(HEADER + BLOCK, WIN_H - BLOCK, BLOCK))
            if pt not in self.p2_snake:
                self.p2_food = pt; break

    # ─────────────────────────────────────────
    # Collision  (hitbox = boundary cell entry)
    # ─────────────────────────────────────────

    def _p1_collision(self, pt: Point) -> bool:
        if pt.x < 0 or pt.x >= DIV_X:       return True   # left wall OR divider
        if pt.y < HEADER or pt.y >= WIN_H:   return True   # top / bottom
        if pt in self.p1_snake[1:]:        return True   # self
        return False

    def _p2_collision(self, pt: Point) -> bool:
        if pt.x < DIV_X or pt.x >= WIN_W:   return True   # divider OR right wall
        if pt.y < HEADER or pt.y >= WIN_H:  return True   # top / bottom
        if pt in self.p2_snake[1:]:          return True   # self
        return False

    def p2_collision_virtual(self, vpt: Point) -> bool:
        """Collision check in model-space coords (real_x = vpt.x + P2_OFFSET)."""
        return self._p2_collision(Point(vpt.x + P2_OFFSET, vpt.y))

    # ─────────────────────────────────────────
    # P2 state vector  (virtual coordinates)
    # ─────────────────────────────────────────

    def get_p2_state(self) -> np.ndarray:
        head  = Point(self.p2_head.x - P2_OFFSET, self.p2_head.y)
        food  = Point(self.p2_food.x - P2_OFFSET, self.p2_food.y)
        dir_  = self.p2_dir

        pt_r = Point(head.x + BLOCK, head.y)
        pt_l = Point(head.x - BLOCK, head.y)
        pt_u = Point(head.x, head.y - BLOCK)
        pt_d = Point(head.x, head.y + BLOCK)

        dr = dir_ == Direction.RIGHT
        dl = dir_ == Direction.LEFT
        du = dir_ == Direction.UP
        dd = dir_ == Direction.DOWN

        def d(vpt): return self.p2_collision_virtual(vpt)

        state = [
            (dr and d(pt_r)) or (dl and d(pt_l)) or (du and d(pt_u)) or (dd and d(pt_d)),
            (du and d(pt_r)) or (dd and d(pt_l)) or (dl and d(pt_u)) or (dr and d(pt_d)),
            (dd and d(pt_r)) or (du and d(pt_l)) or (dr and d(pt_u)) or (dl and d(pt_d)),
            dl, dr, du, dd,
            food.x < head.x, food.x > head.x,
            food.y < head.y, food.y > head.y,
        ]
        return np.array(state, dtype=int)

    # ─────────────────────────────────────────
    # Main step
    # ─────────────────────────────────────────

    def play_step(self, fps: int = 10
                  ) -> tuple[bool, bool, int, int]:
        """
        Advance one frame.  Dead snakes are skipped silently.
        Returns (p1_done, p2_done, p1_score, p2_score).
        Both sides can be done independently.
        """
        self.frame += 1

        # ── Keypress (Both Players) ─────────────────────────────────
        new_dir_p1 = self.p1_dir
        new_dir_p2 = self.p2_dir
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
                
            # Pause menu should work for either player alive
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p and not self.waiting_to_start:
                    self.show_pause_menu()
                    continue
                    
                if self.p1_alive:
                    if event.key in (pygame.K_d,):
                        if self.p1_dir != Direction.LEFT:  new_dir_p1 = Direction.RIGHT
                        self.waiting_to_start = False
                    elif event.key in (pygame.K_a,):
                        if self.p1_dir != Direction.RIGHT: new_dir_p1 = Direction.LEFT
                        self.waiting_to_start = False
                    elif event.key in (pygame.K_w,):
                        if self.p1_dir != Direction.DOWN:  new_dir_p1 = Direction.UP
                        self.waiting_to_start = False
                    elif event.key in (pygame.K_s,):
                        if self.p1_dir != Direction.UP:    new_dir_p1 = Direction.DOWN
                        self.waiting_to_start = False

                if self.p2_alive:
                    if event.key in (pygame.K_RIGHT,):
                        if self.p2_dir != Direction.LEFT:  new_dir_p2 = Direction.RIGHT
                        self.waiting_to_start = False
                    elif event.key in (pygame.K_LEFT,):
                        if self.p2_dir != Direction.RIGHT: new_dir_p2 = Direction.LEFT
                        self.waiting_to_start = False
                    elif event.key in (pygame.K_UP,):
                        if self.p2_dir != Direction.DOWN:  new_dir_p2 = Direction.UP
                        self.waiting_to_start = False
                    elif event.key in (pygame.K_DOWN,):
                        if self.p2_dir != Direction.UP:    new_dir_p2 = Direction.DOWN
                        self.waiting_to_start = False

        # ── Move p1 ────────────────────────────────────────
        if self.p1_alive and not self.waiting_to_start:
            self.p1_dir  = new_dir_p1
            next_pt = _move_pt(self.p1_head, self.p1_dir)
            if self._p1_collision(next_pt) or \
               self.frame > 150 * len(self.p1_snake):
                self.p1_alive = False
            else:
                self.p1_head = next_pt
                self.p1_snake.insert(0, self.p1_head)
                if self.p1_head == self.p1_food:
                    self.p1_score += 1
                    self._place_food_p1()
                else:
                    self.p1_snake.pop()

        # ── Move P2 ───────────────────────────────────────────
        if self.p2_alive and not self.waiting_to_start:
            self.p2_dir  = new_dir_p2
            next_pt = _move_pt(self.p2_head, self.p2_dir)
            if self._p2_collision(next_pt) or \
               self.frame > 150 * len(self.p2_snake):
                self.p2_alive = False
            else:
                self.p2_head = next_pt
                self.p2_snake.insert(0, self.p2_head)
                if self.p2_head == self.p2_food:
                    self.p2_score += 1
                    self._place_food_p2()
                else:
                    self.p2_snake.pop()

        self._render()
        self.clock.tick(fps)

        return (not self.p1_alive), (not self.p2_alive), \
               self.p1_score, self.p2_score

    # ─────────────────────────────────────────
    # Start Screen
    # ─────────────────────────────────────────

    def show_start_screen(self) -> str:
        self.display.fill(C.BG)
        # Draw a fancy background or grid
        self._draw_grid()
        self._draw_divider()

        title_surf = self.font_big.render("GREEN SNAKE VS BLUE SNAKE", True, C.TEXT)
        self.display.blit(title_surf, title_surf.get_rect(center=(WIN_W//2, HEADER + 100)))

        BTN_W, BTN_H, gap = 240, 60, 20
        cx = WIN_W // 2
        mid_y = WIN_H // 2 + 30
        
        btn_run = pygame.Rect(cx - BTN_W//2, mid_y - BTN_H - gap//2, BTN_W, BTN_H)
        btn_lb  = pygame.Rect(cx - BTN_W//2, mid_y + gap//2, BTN_W, BTN_H)

        # Background snake variables
        def reset_bg_snake():
            startX = random.randrange(BLOCK * 3, WIN_W - BLOCK * 3, BLOCK)
            startY = random.randrange(BLOCK * 3, WIN_H - BLOCK * 3, BLOCK)
            return [Point(startX, startY), Point(startX-BLOCK, startY), Point(startX-2*BLOCK, startY)], Direction.RIGHT

        bg_snake, bg_dir = reset_bg_snake()
        bg_food = Point(random.randrange(BLOCK, WIN_W-BLOCK, BLOCK), random.randrange(BLOCK, WIN_H-BLOCK, BLOCK))
        frame_counter = 0

        def greedy_move(head, fd, d):
            dx = fd.x - head.x
            dy = fd.y - head.y
            prefs = []
            if abs(dx) > abs(dy):
                prefs.append(Direction.RIGHT if dx > 0 else Direction.LEFT)
                prefs.append(Direction.DOWN if dy > 0 else Direction.UP)
                prefs.append(Direction.UP if dy > 0 else Direction.DOWN)
                prefs.append(Direction.LEFT if dx > 0 else Direction.RIGHT)
            else:
                prefs.append(Direction.DOWN if dy > 0 else Direction.UP)
                prefs.append(Direction.RIGHT if dx > 0 else Direction.LEFT)
                prefs.append(Direction.LEFT if dx > 0 else Direction.RIGHT)
                prefs.append(Direction.UP if dy > 0 else Direction.DOWN)
                
            for p in prefs:
                n_pt = _move_pt(head, p)
                if n_pt.x >= 0 and n_pt.x < WIN_W and n_pt.y >= 0 and n_pt.y < WIN_H and n_pt not in bg_snake:
                    # also prevent 180 degree turns
                    if (p == Direction.RIGHT and d == Direction.LEFT) or \
                       (p == Direction.LEFT and d == Direction.RIGHT) or \
                       (p == Direction.UP and d == Direction.DOWN) or \
                       (p == Direction.DOWN and d == Direction.UP):
                        continue
                    return p, n_pt
            # if stuck, fallback to current dir
            return d, _move_pt(head, d)

        def draw(hover_run, hover_lb):
            self.display.fill(C.BG)
            # Custom simple grid for full screen start menu
            for x in range(0, WIN_W, BLOCK):
                pygame.draw.line(self.display, C.GRID, (x, 0), (x, WIN_H))
            for y in range(0, WIN_H, BLOCK):
                pygame.draw.line(self.display, C.GRID, (0, y), (WIN_W, y))
                
            # Draw bg snake
            for pt in bg_snake:
                pygame.draw.rect(self.display, C.H_HEAD, pygame.Rect(pt.x, pt.y, BLOCK, BLOCK))
                pygame.draw.rect(self.display, C.BG, pygame.Rect(pt.x, pt.y, BLOCK, BLOCK), 1)
            pygame.draw.rect(self.display, C.FOOD_H, pygame.Rect(bg_food.x, bg_food.y, BLOCK, BLOCK), border_radius=4)
            
            # Draw UI
            self.display.blit(title_surf, title_surf.get_rect(center=(WIN_W//2, HEADER + 100)))

            # Run button
            rc_run = C.BTN_HOV if hover_run else C.BTN_BG
            pygame.draw.rect(self.display, rc_run, btn_run, border_radius=12)
            pygame.draw.rect(self.display, (100,200,255), btn_run, 3, border_radius=12)
            rt = self.font_big.render("RUN", True, C.BTN_TXT)
            self.display.blit(rt, rt.get_rect(center=btn_run.center))
            
            # Leaderboard button
            rc_lb = C.BTN_HOV if hover_lb else C.BTN_BG
            pygame.draw.rect(self.display, rc_lb, btn_lb, border_radius=12)
            pygame.draw.rect(self.display, (240, 200, 100), btn_lb, 3, border_radius=12)
            lbl_t = self.font_btn.render("LEADERBOARD", True, C.BTN_TXT)
            self.display.blit(lbl_t, lbl_t.get_rect(center=btn_lb.center))
            
            pygame.display.flip()

        while True:
            m = pygame.mouse.get_pos()
            
            frame_counter += 1
            if frame_counter % 2 == 0:  # ~15 fps movement speed
                bg_dir, next_pt = greedy_move(bg_snake[0], bg_food, bg_dir)
                if next_pt.x < 0 or next_pt.x >= WIN_W or next_pt.y < 0 or next_pt.y >= WIN_H or next_pt in bg_snake:
                    # Trapped or hit wall; reset snake
                    bg_snake, bg_dir = reset_bg_snake()
                else:
                    bg_snake.insert(0, next_pt)
                    if next_pt == bg_food:
                        # Eat and grow
                        if len(bg_snake) >= 17:
                            bg_snake, bg_dir = reset_bg_snake()
                        else:
                            bg_food = Point(random.randrange(BLOCK, WIN_W-BLOCK, BLOCK), random.randrange(BLOCK, WIN_H-BLOCK, BLOCK))
                    else:
                        bg_snake.pop()
            
            draw(btn_run.collidepoint(m), btn_lb.collidepoint(m))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_run.collidepoint(event.pos): 
                        return "play"
                    elif btn_lb.collidepoint(event.pos):
                        return "leaderboard"
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        return "play"
                    if event.key in (pygame.K_l,):
                        return "leaderboard"
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        pygame.quit(); raise SystemExit
            self.clock.tick(30)

    # ─────────────────────────────────────────
    # Leaderboard Screen
    # ─────────────────────────────────────────

    def show_leaderboard_screen(self) -> None:
        self.display.fill(C.BG)
        self._draw_grid()
        self._draw_divider()
        
        # Titles
        title_hum = self.font_big.render("P1 TOP 10", True, C.H_SCORE)
        title_p2  = self.font_big.render("P2 TOP 10", True, C.P2_SCORE)
        
        self.display.blit(title_hum, title_hum.get_rect(center=(FIELD_W//2, HEADER + 60)))
        self.display.blit(title_p2, title_p2.get_rect(center=(DIV_X + FIELD_W//2, HEADER + 60)))
        
        # Display Scores
        y_start = HEADER + 120
        gap = 30
        
        for i in range(10):
            h_record = self.high_scores["p1"][i] if i < len(self.high_scores["p1"]) else {"score": 0, "name": "-"}
            h_record2 = self.high_scores["p2"][i] if i < len(self.high_scores["p2"]) else {"score": 0, "name": "-"}
            
            h_score = h_record["score"]
            h_name = h_record["name"]
            
            p2_score = h_record2["score"]
            p2_name = h_record2["name"]
            
            # Truncate slightly if name is very long just for display purposes
            disp_name = h_name[:15] + ".." if len(h_name) > 15 else h_name
            h_text_str = f"{i+1}. {disp_name.ljust(15)} {h_score}" if h_score > 0 else f"{i+1}. -"
            
            disp_name2 = p2_name[:15] + ".." if len(p2_name) > 15 else p2_name
            p2_text_str = f"{i+1}. {disp_name2.ljust(15)} {p2_score}" if p2_score > 0 else f"{i+1}. -"
            
            h_txt = self.font_med.render(h_text_str, True, C.TEXT)
            p2_txt = self.font_med.render(p2_text_str, True, C.TEXT)
            
            # center each text string inside its respective field
            self.display.blit(h_txt, h_txt.get_rect(center=(FIELD_W//2, y_start + i * gap)))
            self.display.blit(p2_txt, p2_txt.get_rect(center=(DIV_X + FIELD_W//2, y_start + i * gap)))
            
        BTN_W, BTN_H = 180, 50
        btn_back = pygame.Rect(10, WIN_H - BTN_H - 10, BTN_W, BTN_H)
        
        def draw(hover):
            rc = C.BTN_HOV if hover else C.BTN_BG
            pygame.draw.rect(self.display, rc, btn_back, border_radius=10)
            pygame.draw.rect(self.display, C.BORDER, btn_back, 2, border_radius=10)
            bt = self.font_btn.render("BACK", True, C.BTN_TXT)
            self.display.blit(bt, bt.get_rect(center=btn_back.center))
            pygame.display.flip()
            
        while True:
            m = pygame.mouse.get_pos()
            draw(btn_back.collidepoint(m))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_back.collidepoint(event.pos): return
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE, pygame.K_b):
                        return
            self.clock.tick(30)

    # ─────────────────────────────────────────
    # Pause Menu
    # ─────────────────────────────────────────

    def show_pause_menu(self) -> str:
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        
        BTN_W, BTN_H, gap = 240, 50, 20
        cx = WIN_W // 2
        mid_y = WIN_H // 2
        
        title_surf = self.font_big.render("PAUSED", True, C.TEXT)
        
        btn_c = pygame.Rect(cx - BTN_W // 2, mid_y - BTN_H - gap, BTN_W, BTN_H)
        btn_p = pygame.Rect(cx - BTN_W // 2, mid_y,               BTN_W, BTN_H)
        btn_e = pygame.Rect(cx - BTN_W // 2, mid_y + BTN_H + gap, BTN_W, BTN_H)

        def draw(hc, hp, he):
            self.display.blit(overlay, (0, 0))
            self.display.blit(title_surf, title_surf.get_rect(center=(cx, mid_y - BTN_H*2 - gap*2)))

            # Continue
            cc = C.BTN_HOV if hc else C.BTN_BG
            pygame.draw.rect(self.display, cc, btn_c, border_radius=10)
            pygame.draw.rect(self.display, (100,200,255), btn_c, 2, border_radius=10)
            ct = self.font_btn.render("CONTINUE", True, C.BTN_TXT)
            self.display.blit(ct, ct.get_rect(center=btn_c.center))

            # Play Again
            pc = C.BTN_HOV if hp else C.BTN_BG
            pygame.draw.rect(self.display, pc, btn_p, border_radius=10)
            pygame.draw.rect(self.display, C.H_SCORE, btn_p, 2, border_radius=10)
            pt = self.font_btn.render("PLAY AGAIN", True, C.BTN_TXT)
            self.display.blit(pt, pt.get_rect(center=btn_p.center))

            # Exit
            ec = (180,55,55) if he else (120,35,35)
            pygame.draw.rect(self.display, ec, btn_e, border_radius=10)
            pygame.draw.rect(self.display, (230,90,90), btn_e, 2, border_radius=10)
            et = self.font_btn.render("EXIT", True, (255,210,210))
            self.display.blit(et, et.get_rect(center=btn_e.center))

            pygame.display.flip()

        while True:
            m = pygame.mouse.get_pos()
            draw(btn_c.collidepoint(m), btn_p.collidepoint(m), btn_e.collidepoint(m))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_c.collidepoint(event.pos): return "continue"
                    if btn_p.collidepoint(event.pos): 
                        self.reset()
                        return "restart"
                    if btn_e.collidepoint(event.pos):
                        pygame.quit(); raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_c, pygame.K_p, pygame.K_ESCAPE):
                        return "continue"
                    if event.key == pygame.K_r: 
                        self.reset()
                        return "restart"
                    if event.key == pygame.K_q:
                        pygame.quit(); raise SystemExit
            self.clock.tick(30)

    def get_player_name_input(self, title_text: str = "NEW HIGH SCORE!") -> str:
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))

        title_surf = self.font_big.render(title_text, True, (255, 215, 0))
        sub_surf = self.font_med.render("Enter your name:", True, C.TEXT)
        
        name = ""
        input_rect = pygame.Rect(WIN_W//2 - 150, WIN_H//2, 300, 50)
        color_active = (100, 200, 255)
        color_passive = (150, 150, 150)
        active = True
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return name.strip() or "Anonymous"
                    elif event.key == pygame.K_BACKSPACE:
                        name = name[:-1]
                    else:
                        if len(name) < 20 and event.unicode.isprintable():
                            name += event.unicode
                            
            self.display.blit(overlay, (0, 0))
            self.display.blit(title_surf, title_surf.get_rect(center=(WIN_W//2, WIN_H//2 - 80)))
            self.display.blit(sub_surf, sub_surf.get_rect(center=(WIN_W//2, WIN_H//2 - 30)))
            
            pygame.draw.rect(self.display, color_active, input_rect, 2, border_radius=5)
            
            txt_surface = self.font_med.render(name + "_", True, (255, 255, 255))
            self.display.blit(txt_surface, (input_rect.x + 10, input_rect.y + 13))
            
            pygame.display.flip()
            self.clock.tick(30)

    def show_result_screen(self) -> str:
        # Check P1 High Score
        min_p1 = min([s["score"] for s in self.high_scores["p1"]]) if len(self.high_scores["p1"]) >= 10 else 0
        if self.p1_score > min_p1 or (self.p1_score > 0 and len(self.high_scores["p1"]) < 10):
            p_name = self.get_player_name_input("P1: NEW HIGH SCORE!")
            self.high_scores["p1"].append({"score": self.p1_score, "name": p_name})
            self.high_scores["p1"].sort(key=lambda x: x["score"], reverse=True)
            self.high_scores["p1"] = self.high_scores["p1"][:10]
            
        # Check P2 High Score
        min_p2 = min([s["score"] for s in self.high_scores["p2"]]) if len(self.high_scores["p2"]) >= 10 else 0
        if self.p2_score > min_p2 or (self.p2_score > 0 and len(self.high_scores["p2"]) < 10):
            p_name = self.get_player_name_input("P2: NEW HIGH SCORE!")
            self.high_scores["p2"].append({"score": self.p2_score, "name": p_name})
            self.high_scores["p2"].sort(key=lambda x: x["score"], reverse=True)
            self.high_scores["p2"] = self.high_scores["p2"][:10]
        
        self._save_scores()
        
        if self.p1_score > self.p2_score:
            title, tcol = "P1 WINS!", C.H_SCORE
        elif self.p2_score > self.p1_score:
            title, tcol = "P2 WINS!",   C.P2_SCORE
        else:
            title, tcol = "DRAW!",      (220, 200, 80)

        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))

        BTN_W, BTN_H, gap = 240, 50, 20
        cx = WIN_W // 2
        mid_y = WIN_H // 2

        btn_p = pygame.Rect(cx - BTN_W // 2, mid_y - BTN_H - gap, BTN_W, BTN_H)
        btn_m = pygame.Rect(cx - BTN_W // 2, mid_y,               BTN_W, BTN_H)
        btn_e = pygame.Rect(cx - BTN_W // 2, mid_y + BTN_H + gap, BTN_W, BTN_H)

        t_surf  = self.font_big.render(title, True, tcol)
        sc_surf = self.font_med.render(
            f"P1 {self.p1_score}   P2 {self.p2_score}", True, C.TEXT)

        def draw(hp, hm, he):
            self.display.blit(overlay, (0, 0))
            self.display.blit(t_surf,  t_surf.get_rect(center=(cx, mid_y - BTN_H*2 - gap*2)))
            self.display.blit(sc_surf, sc_surf.get_rect(center=(cx, mid_y - BTN_H*2 - gap//2)))

            # Play Again
            pc = C.BTN_HOV if hp else C.BTN_BG
            pygame.draw.rect(self.display, pc, btn_p, border_radius=10)
            pygame.draw.rect(self.display, C.H_SCORE, btn_p, 2, border_radius=10)
            pt = self.font_btn.render("PLAY AGP2N", True, C.BTN_TXT)
            self.display.blit(pt, pt.get_rect(center=btn_p.center))

            # Menu
            mc = C.BTN_HOV if hm else C.BTN_BG
            pygame.draw.rect(self.display, mc, btn_m, border_radius=10)
            pygame.draw.rect(self.display, (100,200,255), btn_m, 2, border_radius=10)
            mt = self.font_btn.render("MENU", True, C.BTN_TXT)
            self.display.blit(mt, mt.get_rect(center=btn_m.center))

            # Exit
            ec = (180,55,55) if he else (120,35,35)
            pygame.draw.rect(self.display, ec, btn_e, border_radius=10)
            pygame.draw.rect(self.display, (230,90,90), btn_e, 2, border_radius=10)
            et = self.font_btn.render("EXIT", True, (255,210,210))
            self.display.blit(et, et.get_rect(center=btn_e.center))

            pygame.display.flip()

        while True:
            m = pygame.mouse.get_pos()
            draw(btn_p.collidepoint(m), btn_m.collidepoint(m), btn_e.collidepoint(m))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_p.collidepoint(event.pos): return "restart"
                    if btn_m.collidepoint(event.pos): return "menu"
                    if btn_e.collidepoint(event.pos):
                        pygame.quit(); raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: return "restart"
                    if event.key == pygame.K_m: return "menu"
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        pygame.quit(); raise SystemExit
            self.clock.tick(30)

    # ─────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────

    def _render(self) -> None:
        self.display.fill(C.BG)
        self._draw_grid()
        self._draw_food()
        self._draw_snake(self.p1_snake, self.p1_dir, p1=True)
        self._draw_snake(self.p2_snake,    self.p2_dir,    p1=False)
        self._draw_dead_overlays()
        self._draw_divider()
        self._draw_walls()
        self._draw_header()
        pygame.display.flip()

    def _draw_grid(self) -> None:
        for x in range(0, WIN_W, BLOCK):
            pygame.draw.line(self.display, C.GRID, (x, HEADER), (x, WIN_H))
        for y in range(HEADER, WIN_H, BLOCK):
            pygame.draw.line(self.display, C.GRID, (0, y), (WIN_W, y))

    def _draw_divider(self) -> None:
        """Thin blue vertical line — hitbox is at x = DIV_X."""
        pygame.draw.line(self.display, C.DIVIDER,
                         (DIV_X, HEADER), (DIV_X, WIN_H), DIV_VIS)

    def _draw_walls(self) -> None:
        pygame.draw.rect(self.display, C.WALL,
                         pygame.Rect(0, HEADER, WIN_W, WIN_H - HEADER), 3)

    def _draw_food(self) -> None:
        for food, col in ((self.p1_food, C.FOOD_H), (self.p2_food, C.FOOD_P2)):
            r = pygame.Rect(food.x, food.y, BLOCK, BLOCK)
            pygame.draw.rect(self.display, col, r, border_radius=10)
            shine = pygame.Rect(food.x+4, food.y+3, 5, 4)
            pygame.draw.ellipse(self.display, tuple(min(c+60, 255) for c in col), shine)

    def _draw_snake(self, snake, direction, p1: bool) -> None:
        H = (C.H_HEAD,  C.H_BODY1,  C.H_BODY2)  if p1 else \
            (C.P2_HEAD, C.P2_BODY1, C.P2_BODY2)
        for i, pt in enumerate(snake):
            r = pygame.Rect(pt.x, pt.y, BLOCK, BLOCK)
            if i == 0:
                pygame.draw.rect(self.display, H[0], r, border_radius=6)
                eye_r = 3
                if   direction == Direction.RIGHT: eyes = [(pt.x+BLOCK-6, pt.y+4), (pt.x+BLOCK-6, pt.y+BLOCK-7)]
                elif direction == Direction.LEFT:  eyes = [(pt.x+5, pt.y+4), (pt.x+5, pt.y+BLOCK-7)]
                elif direction == Direction.UP:    eyes = [(pt.x+4, pt.y+5), (pt.x+BLOCK-7, pt.y+5)]
                else:                              eyes = [(pt.x+4, pt.y+BLOCK-6), (pt.x+BLOCK-7, pt.y+BLOCK-6)]
                for ex, ey in eyes:
                    pygame.draw.circle(self.display, (0,0,0),       (ex,ey), eye_r)
                    pygame.draw.circle(self.display, (255,255,255), (ex-1,ey-1), 1)
            else:
                col = H[1] if i % 2 == 0 else H[2]
                pygame.draw.rect(self.display, col, r, border_radius=4)
                inner = pygame.Rect(pt.x+3, pt.y+3, BLOCK-6, BLOCK-6)
                pygame.draw.rect(self.display, tuple(min(c+15,255) for c in col), inner, border_radius=2)

    def _draw_dead_overlays(self) -> None:
        """Dim a side and print ELIMINATED when that snake has died."""
        if not self.p1_alive:
            self.display.blit(self._dead_surf_l, (0, HEADER))
            txt = self.font_elim.render("ELIMINATED", True, (200, 80, 80))
            self.display.blit(txt, txt.get_rect(center=(FIELD_W // 2, WIN_H // 2)))
        if not self.p2_alive:
            self.display.blit(self._dead_surf_r, (DIV_X, HEADER))
            txt = self.font_elim.render("ELIMINATED", True, (80, 140, 220))
            self.display.blit(txt, txt.get_rect(center=(DIV_X + FIELD_W // 2, WIN_H // 2)))

    def _draw_header(self) -> None:
        pygame.draw.rect(self.display, C.HEADER_BG, (0, 0, WIN_W, HEADER))
        pygame.draw.line(self.display, C.BORDER, (0, HEADER), (WIN_W, HEADER), 1)
        pygame.draw.line(self.display, C.DIVIDER, (DIV_X, 0), (DIV_X, HEADER), 2)

        h_lbl = self.font_label.render("P1", True, C.H_SCORE)
        h_scr = self.font_score.render(f"{self.p1_score:>3}", True, C.H_SCORE)
        self.display.blit(h_lbl, (10, 6))
        self.display.blit(h_scr, (10 + h_lbl.get_width() + 8,
                                  (HEADER - h_scr.get_height()) // 2))

        p2_scr = self.font_score.render(f"{self.p2_score:>3}", True, C.P2_SCORE)
        p2_lbl = self.font_label.render("P2",    True, C.P2_SCORE)
        rx = WIN_W - p2_scr.get_width() - p2_lbl.get_width() - 18
        self.display.blit(p2_scr, (rx, (HEADER - p2_scr.get_height()) // 2))
        self.display.blit(p2_lbl, (rx + p2_scr.get_width() + 8, 6))

        vs = self.font_label.render("VS", True, C.DIVIDER)
        self.display.blit(vs, vs.get_rect(center=(DIV_X, HEADER // 2)))
