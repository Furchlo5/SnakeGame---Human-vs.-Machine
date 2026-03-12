"""
Snake Game — RL-ready implementation
=====================================
Author: Antigravity

Two modes:
  AI_PLAYING = True  → external agent calls play_step(action) each tick
  AI_PLAYING = False → human controls with arrow / WASD keys
"""

import pygame
import random
from enum import Enum
from collections import namedtuple

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
BLOCK     = 20          # pixel size of one grid cell
GRID_W    = 30          # grid columns  → window width  = 600 px
GRID_H    = 25          # grid rows     → window height = 500 px + header bar
HEADER    = 40          # pixels reserved for the score bar
WIN_W     = BLOCK * GRID_W
WIN_H     = BLOCK * GRID_H + HEADER

SPEED_HUMAN = 12        # FPS for human mode
# AI mode ignores FPS cap (clock.tick() called with 0)

# ──────────────────────────────────────────────
# Colours
# ──────────────────────────────────────────────
class C:
    BG        = (15,  15,  25)
    GRID      = (25,  25,  40)
    HEADER_BG = (10,  10,  20)
    SNAKE_H   = (80,  220,  80)   # head — bright green
    SNAKE_B1  = (40,  160,  40)   # body — mid green
    SNAKE_B2  = (25,  110,  25)   # body — dark green (checker)
    FOOD      = (220,  55,  55)   # red
    FOOD_SH   = (160,  20,  20)   # food shadow / shine
    TEXT      = (200, 200, 200)
    SCORE_HL  = (120, 230, 120)
    BORDER    = (50,   50,  80)
    WALL      = (70,   70, 160)   # wall blocks (filled cells)
    WALL_HL   = (110, 110, 210)   # wall highlight / bevel

# ──────────────────────────────────────────────
# Direction
# ──────────────────────────────────────────────
class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4

Point = namedtuple('Point', ['x', 'y'])

# ──────────────────────────────────────────────
# Action encoding (relative)
# ──────────────────────────────────────────────
# action = [1, 0, 0] → straight
# action = [0, 1, 0] → turn right (clockwise)
# action = [0, 0, 1] → turn left  (counter-clockwise)

_CLOCK_ORDER = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


def _resolve_direction(current: Direction, action: list[int]) -> Direction:
    idx = _CLOCK_ORDER.index(current)
    if action[1]:          # right turn
        idx = (idx + 1) % 4
    elif action[2]:        # left turn
        idx = (idx - 1) % 4
    # action[0] → straight, no change
    return _CLOCK_ORDER[idx]


# ──────────────────────────────────────────────
# SnakeGameAI
# ──────────────────────────────────────────────
class SnakeGameAI:
    """
    Encapsulates the Snake environment.

    Parameters
    ----------
    ai_playing : bool
        True  → designed for an RL agent; no FPS cap, no keyboard input.
        False → human-playable; arrow / WASD keys, capped at SPEED_HUMAN FPS.
    render : bool
        True  → shows a Pygame window.
        False → headless mode (no window); useful for ultra-fast training.
    """

    def __init__(self, ai_playing: bool = True, render: bool = True):
        self.ai_playing = ai_playing
        self.do_render  = render

        pygame.init()
        if self.do_render:
            pygame.display.set_caption("Snake — RL Edition")
            self.display = pygame.display.set_mode((WIN_W, WIN_H))
            self.font_score  = pygame.font.SysFont("monospace",   20, bold=True)
            self.font_label  = pygame.font.SysFont("monospace",   13)
        else:
            # Headless: we still need the display surface for logic checks
            # but we never call pygame.display.flip()
            self.display = pygame.Surface((WIN_W, WIN_H))

        self.clock = pygame.time.Clock()

        # Game state initialised by reset()
        self.direction: Direction = Direction.RIGHT
        self.head:      Point     = Point(0, 0)
        self.snake:     list[Point] = []
        self.score:     int       = 0
        self.food:      Point     = Point(0, 0)
        self.frame_iteration: int = 0   # steps since last reset

        self.reset()

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def reset(self) -> None:
        """Restart the environment to its initial state."""
        self.direction = Direction.RIGHT

        cx, cy = WIN_W // 2, (WIN_H - HEADER) // 2 + HEADER
        # Snap to grid
        cx = (cx // BLOCK) * BLOCK
        cy = (cy // BLOCK) * BLOCK

        self.head  = Point(cx, cy)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK,     self.head.y),
            Point(self.head.x - 2 * BLOCK, self.head.y),
        ]
        self.score          = 0
        self.frame_iteration = 0
        self._place_food()

    def play_step(self, action: list[int]) -> tuple[float, bool, int]:
        """
        Advance the game by one step.

        Parameters
        ----------
        action : list[int]
            One-hot encoded relative action: [straight, right, left].

        Returns
        -------
        reward : float
        game_over : bool
        score : int
        """
        self.frame_iteration += 1

        # 1. Process OS events (keep window alive and allow quitting)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # 2. Move
        self.direction = _resolve_direction(self.direction, action)
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # 3. Check collision / stuck loop
        reward: float = 0.0
        game_over      = False

        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            reward    = -10.0
            game_over = True
            if self.do_render:
                self._render()
            return reward, game_over, self.score

        # 4. Check food
        if self.head == self.food:
            self.score += 1
            reward      = 10.0
            self._place_food()
        else:
            self.snake.pop()   # remove tail
            reward = -0.01     # tiny penalty per step — discourages circling

        # 5. Render
        if self.do_render:
            self._render()

        # 6. Frame-rate control
        if not self.ai_playing:
            self.clock.tick(SPEED_HUMAN)
        else:
            self.clock.tick(0)   # uncapped — as fast as the CPU goes

        return reward, game_over, self.score

    def play_human_step(self, fps: int = SPEED_HUMAN) -> tuple[bool, int]:
        """
        Advance the game by one step using keyboard input (human mode).

        Parameters
        ----------
        fps : int
            Frame rate cap for this step. Pass a dynamic value to vary speed.

        Returns
        -------
        game_over : bool
        score : int
        """
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    if self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    if self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                elif event.key in (pygame.K_UP, pygame.K_w):
                    if self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    if self.direction != Direction.UP:
                        self.direction = Direction.DOWN

        self._move(self.direction)
        self.snake.insert(0, self.head)

        game_over = False
        if self._is_collision():
            game_over = True
            if self.do_render:
                self._render()
                self.clock.tick(fps)
            return game_over, self.score

        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        if self.do_render:
            self._render()
        self.clock.tick(fps)

        return game_over, self.score

    # ─────────────────────────────────────────
    # State accessors (useful for RL agents)
    # ─────────────────────────────────────────

    @property
    def head_pos(self) -> Point:
        return self.head

    @property
    def body(self) -> list[Point]:
        """Snake body (excluding head)."""
        return self.snake[1:]

    @property
    def current_direction(self) -> Direction:
        return self.direction

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _place_food(self) -> None:
        while True:
            fx = random.randrange(0, WIN_W, BLOCK)
            fy = random.randrange(HEADER, WIN_H, BLOCK)
            candidate = Point(fx, fy)
            if candidate not in self.snake:
                self.food = candidate
                break

    def _move(self, direction: Direction) -> None:
        x, y = self.head
        if   direction == Direction.RIGHT: x += BLOCK
        elif direction == Direction.LEFT:  x -= BLOCK
        elif direction == Direction.UP:    y -= BLOCK
        elif direction == Direction.DOWN:  y += BLOCK
        self.head = Point(x, y)

    def _is_collision(self, pt: Point | None = None) -> bool:
        if pt is None:
            pt = self.head
        # Wall collision — true window boundary
        if pt.x < 0 or pt.x >= WIN_W:
            return True
        if pt.y < HEADER or pt.y >= WIN_H:
            return True
        # Self collision
        if pt in self.snake[1:]:
            return True
        return False

    def is_collision(self, pt: Point | None = None) -> bool:
        """Public wrapper — for use by RL state builders."""
        return self._is_collision(pt)

    def show_game_over_screen(self) -> str:
        """
        Draw a Game Over overlay with Restart and Exit buttons.

        Returns 'restart' or raises SystemExit on Exit / window close.
        Keyboard shortcuts: R = restart, Q / Esc = exit.
        """
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 175))
        self.display.blit(overlay, (0, 0))

        cx = WIN_W // 2

        font_big = pygame.font.SysFont("monospace", 46, bold=True)
        font_med = pygame.font.SysFont("monospace", 20, bold=True)
        font_btn = pygame.font.SysFont("monospace", 18, bold=True)

        title      = font_big.render("GAME OVER", True, (220, 60, 60))
        score_surf = font_med.render(f"Score: {self.score}", True, (180, 230, 180))

        BTN_W, BTN_H = 200, 46
        gap = 16
        btn_restart = pygame.Rect(cx - BTN_W // 2, WIN_H // 2 + 10, BTN_W, BTN_H)
        btn_exit    = pygame.Rect(cx - BTN_W // 2, WIN_H // 2 + 10 + BTN_H + gap, BTN_W, BTN_H)

        def draw(hover_r: bool, hover_e: bool) -> None:
            self.display.blit(overlay, (0, 0))
            self.display.blit(title,      title.get_rect(center=(cx, WIN_H // 2 - 80)))
            self.display.blit(score_surf, score_surf.get_rect(center=(cx, WIN_H // 2 - 30)))

            rc = (70, 190, 70) if hover_r else (45, 130, 45)
            pygame.draw.rect(self.display, rc, btn_restart, border_radius=10)
            pygame.draw.rect(self.display, (120, 230, 120), btn_restart, 2, border_radius=10)
            rt = font_btn.render("  RESTART", True, (230, 255, 230))
            self.display.blit(rt, rt.get_rect(center=btn_restart.center))

            ec = (180, 55, 55) if hover_e else (120, 35, 35)
            pygame.draw.rect(self.display, ec, btn_exit, border_radius=10)
            pygame.draw.rect(self.display, (230, 90, 90), btn_exit, 2, border_radius=10)
            et = font_btn.render("  EXIT", True, (255, 210, 210))
            self.display.blit(et, et.get_rect(center=btn_exit.center))

            pygame.display.flip()

        while True:
            m = pygame.mouse.get_pos()
            draw(btn_restart.collidepoint(m), btn_exit.collidepoint(m))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_restart.collidepoint(event.pos):
                        return "restart"
                    if btn_exit.collidepoint(event.pos):
                        pygame.quit()
                        raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        return "restart"
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        pygame.quit()
                        raise SystemExit

            self.clock.tick(30)

    # ─────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────

    def _render(self) -> None:
        self.display.fill(C.BG)
        self._draw_grid()
        self._draw_food()
        self._draw_snake()
        self._draw_walls()   # drawn on top so snake can't visually "escape"
        self._draw_header()
        pygame.display.flip()

    def _draw_grid(self) -> None:
        """Subtle grid lines."""
        for x in range(0, WIN_W, BLOCK):
            pygame.draw.line(self.display, C.GRID, (x, HEADER), (x, WIN_H))
        for y in range(HEADER, WIN_H, BLOCK):
            pygame.draw.line(self.display, C.GRID, (0, y), (WIN_W, y))

    def _draw_walls(self) -> None:
        """Draw a thin border around the play area."""
        play_rect = pygame.Rect(0, HEADER, WIN_W, WIN_H - HEADER)
        pygame.draw.rect(self.display, C.WALL, play_rect, 3)

    def _draw_snake(self) -> None:
        for i, pt in enumerate(self.snake):
            r = pygame.Rect(pt.x, pt.y, BLOCK, BLOCK)
            if i == 0:
                # Head — bright rounded rect
                pygame.draw.rect(self.display, C.SNAKE_H, r, border_radius=6)
                # Eyes
                eye_r = 3
                if self.direction == Direction.RIGHT:
                    eyes = [(pt.x + BLOCK - 6, pt.y + 4),
                            (pt.x + BLOCK - 6, pt.y + BLOCK - 7)]
                elif self.direction == Direction.LEFT:
                    eyes = [(pt.x + 5, pt.y + 4),
                            (pt.x + 5, pt.y + BLOCK - 7)]
                elif self.direction == Direction.UP:
                    eyes = [(pt.x + 4,          pt.y + 5),
                            (pt.x + BLOCK - 7,  pt.y + 5)]
                else:
                    eyes = [(pt.x + 4,          pt.y + BLOCK - 6),
                            (pt.x + BLOCK - 7,  pt.y + BLOCK - 6)]
                for ex, ey in eyes:
                    pygame.draw.circle(self.display, (0, 0, 0), (ex, ey), eye_r)
                    pygame.draw.circle(self.display, (255, 255, 255), (ex - 1, ey - 1), 1)
            else:
                colour = C.SNAKE_B1 if i % 2 == 0 else C.SNAKE_B2
                pygame.draw.rect(self.display, colour, r, border_radius=4)
                # Inner highlight
                inner = pygame.Rect(pt.x + 3, pt.y + 3, BLOCK - 6, BLOCK - 6)
                pygame.draw.rect(self.display, tuple(min(c + 15, 255) for c in colour), inner, border_radius=2)

    def _draw_food(self) -> None:
        r = pygame.Rect(self.food.x, self.food.y, BLOCK, BLOCK)
        pygame.draw.rect(self.display, C.FOOD,    r, border_radius=10)
        # Shine spot
        shine = pygame.Rect(self.food.x + 4, self.food.y + 3, 5, 4)
        pygame.draw.ellipse(self.display, (255, 150, 150), shine)

    def _draw_header(self) -> None:
        # Background bar
        pygame.draw.rect(self.display, C.HEADER_BG, (0, 0, WIN_W, HEADER))
        pygame.draw.line(self.display, C.BORDER, (0, HEADER), (WIN_W, HEADER), 1)

        # Title
        title = self.font_label.render("SNAKE  RL", True, C.TEXT)
        self.display.blit(title, (10, (HEADER - title.get_height()) // 2))

        # Score
        score_txt = self.font_score.render(f"SCORE  {self.score:>4}", True, C.SCORE_HL)
        self.display.blit(score_txt, (WIN_W - score_txt.get_width() - 10,
                                      (HEADER - score_txt.get_height()) // 2))

        # Mode badge
        mode     = "AI" if self.ai_playing else "HUMAN"
        badge_c  = (60, 160, 220) if self.ai_playing else (220, 160, 60)
        badge    = self.font_label.render(f"[{mode}]", True, badge_c)
        self.display.blit(badge, (WIN_W // 2 - badge.get_width() // 2,
                                  (HEADER - badge.get_height()) // 2))
