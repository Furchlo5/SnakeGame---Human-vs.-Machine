"""
hvai_game.py — Human vs. AI dual-field Snake game
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
Hitbox : Human dies at x >= 500 | AI dies at x < 500
         (snake enters the cell containing the divider → game over)

AI virtual coordinates
----------------------
Model sees x in [0, 500) — same as its training space.
Real AI x is in [500, 1000).  offset = 500.
  virtual_x = real_x - 500
"""

import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

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

AI_OFFSET = DIV_X      # subtract from real AI x → virtual x for model


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

    AI_HEAD  = (80,  180, 255)
    AI_BODY1 = (40,  110, 200)
    AI_BODY2 = (25,   70, 150)

    FOOD_H  = (220,  55,  55)
    FOOD_AI = (255, 165,   0)

    TEXT     = (200, 200, 200)
    DEAD_DIM = (0,   0,   0,  140)   # SRCALPHA overlay for dead side
    H_SCORE  = (120, 230, 120)
    AI_SCORE = (100, 180, 255)
    BORDER   = (50,  50,  80)


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
# HumanVsAIGame
# ──────────────────────────────────────────────
class HumanVsAIGame:
    """
    Dual-field Snake.

    Each side plays independently.  When one snake dies its side dims and
    shows "ELIMINATED" but the other continues.  Game ends only when both
    are dead, then show_result_screen() is called.
    """

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake — Human vs. AI")
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

        self.reset()

    # ─────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────

    def reset(self) -> None:
        self.frame = 0

        # Human snake — centre of left field
        hx = (FIELD_W // 2 // BLOCK) * BLOCK           # 240
        hy = (FIELD_H // 2 // BLOCK) * BLOCK + HEADER  # 260
        self.human_dir   = Direction.RIGHT
        self.human_snake = [Point(hx, hy), Point(hx-BLOCK, hy), Point(hx-2*BLOCK, hy)]
        self.human_head  = self.human_snake[0]
        self.human_score = 0
        self.human_alive = True

        # AI snake — centre of right field
        ax = DIV_X + (FIELD_W // 2 // BLOCK) * BLOCK   # 740
        self.ai_dir   = Direction.RIGHT
        self.ai_snake = [Point(ax, hy), Point(ax-BLOCK, hy), Point(ax-2*BLOCK, hy)]
        self.ai_head  = self.ai_snake[0]
        self.ai_score = 0
        self.ai_alive = True

        self._place_food_human()
        self._place_food_ai()

    # ─────────────────────────────────────────
    # Food
    # ─────────────────────────────────────────

    def _place_food_human(self) -> None:
        while True:
            pt = Point(random.randrange(BLOCK, FIELD_W - BLOCK, BLOCK),
                       random.randrange(HEADER + BLOCK, WIN_H - BLOCK, BLOCK))
            if pt not in self.human_snake:
                self.human_food = pt; break

    def _place_food_ai(self) -> None:
        while True:
            pt = Point(random.randrange(DIV_X + BLOCK, WIN_W - BLOCK, BLOCK),
                       random.randrange(HEADER + BLOCK, WIN_H - BLOCK, BLOCK))
            if pt not in self.ai_snake:
                self.ai_food = pt; break

    # ─────────────────────────────────────────
    # Collision  (hitbox = boundary cell entry)
    # ─────────────────────────────────────────

    def _human_collision(self, pt: Point) -> bool:
        if pt.x < 0 or pt.x >= DIV_X:       return True   # left wall OR divider
        if pt.y < HEADER or pt.y >= WIN_H:   return True   # top / bottom
        if pt in self.human_snake[1:]:        return True   # self
        return False

    def _ai_collision(self, pt: Point) -> bool:
        if pt.x < DIV_X or pt.x >= WIN_W:   return True   # divider OR right wall
        if pt.y < HEADER or pt.y >= WIN_H:  return True   # top / bottom
        if pt in self.ai_snake[1:]:          return True   # self
        return False

    def ai_collision_virtual(self, vpt: Point) -> bool:
        """Collision check in model-space coords (real_x = vpt.x + AI_OFFSET)."""
        return self._ai_collision(Point(vpt.x + AI_OFFSET, vpt.y))

    # ─────────────────────────────────────────
    # AI state vector  (virtual coordinates)
    # ─────────────────────────────────────────

    def get_ai_state(self) -> np.ndarray:
        head  = Point(self.ai_head.x - AI_OFFSET, self.ai_head.y)
        food  = Point(self.ai_food.x - AI_OFFSET, self.ai_food.y)
        dir_  = self.ai_dir

        pt_r = Point(head.x + BLOCK, head.y)
        pt_l = Point(head.x - BLOCK, head.y)
        pt_u = Point(head.x, head.y - BLOCK)
        pt_d = Point(head.x, head.y + BLOCK)

        dr = dir_ == Direction.RIGHT
        dl = dir_ == Direction.LEFT
        du = dir_ == Direction.UP
        dd = dir_ == Direction.DOWN

        def d(vpt): return self.ai_collision_virtual(vpt)

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

    def play_step(self, ai_action: list[int], fps: int = 10
                  ) -> tuple[bool, bool, int, int]:
        """
        Advance one frame.  Dead snakes are skipped silently.
        Returns (human_done, ai_done, human_score, ai_score).
        Both sides can be done independently.
        """
        self.frame += 1

        # ── Keypress (human) ─────────────────────────────────
        new_dir = self.human_dir
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            if event.type == pygame.KEYDOWN and self.human_alive:
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    if self.human_dir != Direction.LEFT:  new_dir = Direction.RIGHT
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    if self.human_dir != Direction.RIGHT: new_dir = Direction.LEFT
                elif event.key in (pygame.K_UP, pygame.K_w):
                    if self.human_dir != Direction.DOWN:  new_dir = Direction.UP
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    if self.human_dir != Direction.UP:    new_dir = Direction.DOWN

        # ── Move human ────────────────────────────────────────
        if self.human_alive:
            self.human_dir  = new_dir
            self.human_head = _move_pt(self.human_head, self.human_dir)
            self.human_snake.insert(0, self.human_head)
            if self._human_collision(self.human_head) or \
               self.frame > 150 * len(self.human_snake):
                self.human_alive = False
                self.human_snake.pop()
            elif self.human_head == self.human_food:
                self.human_score += 1
                self._place_food_human()
            else:
                self.human_snake.pop()

        # ── Move AI ───────────────────────────────────────────
        if self.ai_alive:
            self.ai_dir  = _resolve_dir(self.ai_dir, ai_action)
            self.ai_head = _move_pt(self.ai_head, self.ai_dir)
            self.ai_snake.insert(0, self.ai_head)
            if self._ai_collision(self.ai_head) or \
               self.frame > 150 * len(self.ai_snake):
                self.ai_alive = False
                self.ai_snake.pop()
            elif self.ai_head == self.ai_food:
                self.ai_score += 1
                self._place_food_ai()
            else:
                self.ai_snake.pop()

        self._render()
        self.clock.tick(fps)

        return (not self.human_alive), (not self.ai_alive), \
               self.human_score, self.ai_score

    # ─────────────────────────────────────────
    # Result screen  (called when BOTH are dead)
    # ─────────────────────────────────────────

    def show_result_screen(self) -> str:
        if self.human_score > self.ai_score:
            title, tcol = "HUMAN WINS!", C.H_SCORE
        elif self.ai_score > self.human_score:
            title, tcol = "AI WINS!",   C.AI_SCORE
        else:
            title, tcol = "DRAW!",      (220, 200, 80)

        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))

        BTN_W, BTN_H, gap = 220, 46, 14
        cx = WIN_W // 2
        mid_y = WIN_H // 2

        btn_r = pygame.Rect(cx - BTN_W // 2, mid_y + 14,                 BTN_W, BTN_H)
        btn_e = pygame.Rect(cx - BTN_W // 2, mid_y + 14 + BTN_H + gap,   BTN_W, BTN_H)

        t_surf  = self.font_big.render(title, True, tcol)
        sc_surf = self.font_med.render(
            f"Human {self.human_score}   AI {self.ai_score}", True, C.TEXT)

        def draw(hr, he):
            self.display.blit(overlay, (0, 0))
            self.display.blit(t_surf,  t_surf.get_rect(center=(cx, mid_y - 65)))
            self.display.blit(sc_surf, sc_surf.get_rect(center=(cx, mid_y - 15)))

            rc = (70,190,70) if hr else (45,130,45)
            pygame.draw.rect(self.display, rc,          btn_r, border_radius=10)
            pygame.draw.rect(self.display, (120,230,120), btn_r, 2, border_radius=10)
            rt = self.font_btn.render("  PLAY AGAIN", True, (230,255,230))
            self.display.blit(rt, rt.get_rect(center=btn_r.center))

            ec = (180,55,55) if he else (120,35,35)
            pygame.draw.rect(self.display, ec,            btn_e, border_radius=10)
            pygame.draw.rect(self.display, (230,90,90),   btn_e, 2, border_radius=10)
            et = self.font_btn.render("  EXIT", True, (255,210,210))
            self.display.blit(et, et.get_rect(center=btn_e.center))

            pygame.display.flip()

        while True:
            m = pygame.mouse.get_pos()
            draw(btn_r.collidepoint(m), btn_e.collidepoint(m))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_r.collidepoint(event.pos): return "restart"
                    if btn_e.collidepoint(event.pos):
                        pygame.quit(); raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: return "restart"
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
        self._draw_snake(self.human_snake, self.human_dir, human=True)
        self._draw_snake(self.ai_snake,    self.ai_dir,    human=False)
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
        for food, col in ((self.human_food, C.FOOD_H), (self.ai_food, C.FOOD_AI)):
            r = pygame.Rect(food.x, food.y, BLOCK, BLOCK)
            pygame.draw.rect(self.display, col, r, border_radius=10)
            shine = pygame.Rect(food.x+4, food.y+3, 5, 4)
            pygame.draw.ellipse(self.display, tuple(min(c+60, 255) for c in col), shine)

    def _draw_snake(self, snake, direction, human: bool) -> None:
        H = (C.H_HEAD,  C.H_BODY1,  C.H_BODY2)  if human else \
            (C.AI_HEAD, C.AI_BODY1, C.AI_BODY2)
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
        if not self.human_alive:
            self.display.blit(self._dead_surf_l, (0, HEADER))
            txt = self.font_elim.render("ELIMINATED", True, (200, 80, 80))
            self.display.blit(txt, txt.get_rect(center=(FIELD_W // 2, WIN_H // 2)))
        if not self.ai_alive:
            self.display.blit(self._dead_surf_r, (DIV_X, HEADER))
            txt = self.font_elim.render("ELIMINATED", True, (80, 140, 220))
            self.display.blit(txt, txt.get_rect(center=(DIV_X + FIELD_W // 2, WIN_H // 2)))

    def _draw_header(self) -> None:
        pygame.draw.rect(self.display, C.HEADER_BG, (0, 0, WIN_W, HEADER))
        pygame.draw.line(self.display, C.BORDER, (0, HEADER), (WIN_W, HEADER), 1)
        pygame.draw.line(self.display, C.DIVIDER, (DIV_X, 0), (DIV_X, HEADER), 2)

        h_lbl = self.font_label.render("HUMAN", True, C.H_SCORE)
        h_scr = self.font_score.render(f"{self.human_score:>3}", True, C.H_SCORE)
        self.display.blit(h_lbl, (10, 6))
        self.display.blit(h_scr, (10 + h_lbl.get_width() + 8,
                                  (HEADER - h_scr.get_height()) // 2))

        ai_scr = self.font_score.render(f"{self.ai_score:>3}", True, C.AI_SCORE)
        ai_lbl = self.font_label.render("AI",    True, C.AI_SCORE)
        rx = WIN_W - ai_scr.get_width() - ai_lbl.get_width() - 18
        self.display.blit(ai_scr, (rx, (HEADER - ai_scr.get_height()) // 2))
        self.display.blit(ai_lbl, (rx + ai_scr.get_width() + 8, 6))

        vs = self.font_label.render("VS", True, C.DIVIDER)
        self.display.blit(vs, vs.get_rect(center=(DIV_X, HEADER // 2)))
