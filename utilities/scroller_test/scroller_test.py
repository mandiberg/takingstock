
import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pygame


CLASSES = {
    80: "Sign",
    81: "Gift",
    82: "Money",
    83: "Bag",
    84: "Valentine",
    85: "Salad",
    86: "Dumbbell",
    87: "Flag",
    88: "Groceries",
    89: "Chestpiece",
    90: "Stethoscope",
    91: "Gun",
    92: "Headphones",
    93: "Clipboard",
    94: "Piggybank",
    95: "Creditcard",
    96: "Bitcoin",
    97: "Rose",
    98: "Lily",
    99: "Iris",
    100: "Tulip",
    101: "Lisianthus",
    102: "Orchid",
    103: "Peony",
    104: "Sunflower",
    105: "Daisy",
    106: "Daffodil",
    107: "Hydrangea",
    108: "Pistol",
    109: "Rifle",
    110: "Mask",
    111: "Facial",
    112: "Sheetmask",
    113: "Eyepatch",
    114: "Sleepmask",
    115: "Masquerade_mask",
    116: "Cucumber",
    117: "Kiwi",
    118: "Lemon_slice",
    119: "Avocado_half",
}


SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 1920
ROWS_PER_SECOND = 10.0
ROW_HEIGHT = 54
TOP_MARGIN = 84
BOTTOM_MARGIN = 64

BG_COLOR = (8, 9, 12)
GRID_COLOR = (32, 35, 42)
TEXT_COLOR = (228, 233, 242)
MUTED_TEXT = (136, 146, 162)
ID_COLOR = (153, 209, 255)
HEADER_COLOR = (166, 179, 199)

LEFT_X = 28
LEFT_W = 430
MID_X = LEFT_X + LEFT_W + 16
MID_W = 190
RIGHT_X = MID_X + MID_W + 16
RIGHT_W = SCREEN_WIDTH - RIGHT_X - 28


@dataclass
class MetaRow:
    objects_text: str
    image_id: str
    description: str


def parse_detection_json(raw_json: str) -> List[dict]:
    if not raw_json:
        return []
    try:
        detections = json.loads(raw_json)
        if isinstance(detections, list):
            return detections
    except Exception:
        pass

    # Some rows may contain Python-literal shaped values.
    try:
        detections = ast.literal_eval(raw_json)
        if isinstance(detections, list):
            return detections
    except Exception:
        return []
    return []


def class_label(class_id: int) -> str:
    return CLASSES.get(class_id, f"class_{class_id}")


def render_detection_text(detections: List[dict]) -> str:
    if not detections:
        return "none"
    parts = []
    for det in detections:
        class_id = det.get("class_id")
        obj_no = det.get("obj_no")
        if class_id is None:
            continue
        if obj_no is None:
            parts.append(f"{class_id}:{class_label(class_id)}")
        else:
            parts.append(f"{class_id}:{class_label(class_id)}#{obj_no}")
    return " | ".join(parts) if parts else "none"


def load_rows(csv_path: Path) -> List[MetaRow]:
    rows: List[MetaRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for raw in reader:
            if not raw:
                continue

            # Headerless schema:
            # image_id, description, topic_score, detection_count, detections_json
            image_id = str(raw[0]).strip() if len(raw) > 0 else ""
            description = str(raw[1]).strip() if len(raw) > 1 else ""
            detections_json = raw[4] if len(raw) > 4 else "[]"

            detections = parse_detection_json(detections_json)
            objects_text = render_detection_text(detections)
            rows.append(MetaRow(objects_text=objects_text, image_id=image_id, description=description))
    return rows


def draw_text_clipped(surface, font, text, color, x, y, width):
    text_surface = font.render(text, True, color)
    if text_surface.get_width() <= width:
        surface.blit(text_surface, (x, y))
        return
    clip_surface = pygame.Surface((width, text_surface.get_height()), pygame.SRCALPHA)
    clip_surface.blit(text_surface, (0, 0))
    surface.blit(clip_surface, (x, y))


def draw_text_fade_right(surface, font, text, color, x, y, width, fade_width, bg_color):
    text_surface = font.render(text, True, color)
    text_h = text_surface.get_height()

    if text_surface.get_width() <= width:
        surface.blit(text_surface, (x, y))
        return

    clip_surface = pygame.Surface((width, text_h), pygame.SRCALPHA)
    clip_surface.blit(text_surface, (0, 0))

    fade_surface = pygame.Surface((fade_width, text_h), pygame.SRCALPHA)
    for i in range(fade_width):
        alpha = int(255 * (i / max(1, fade_width - 1)))
        pygame.draw.line(
            fade_surface,
            (bg_color[0], bg_color[1], bg_color[2], alpha),
            (i, 0),
            (i, text_h),
        )
    clip_surface.blit(fade_surface, (width - fade_width, 0))
    surface.blit(clip_surface, (x, y))


def draw_headers(surface, font):
    pygame.draw.line(surface, GRID_COLOR, (LEFT_X, TOP_MARGIN - 16), (SCREEN_WIDTH - 28, TOP_MARGIN - 16), 1)
    surface.blit(font.render("objects", True, HEADER_COLOR), (LEFT_X, 24))
    surface.blit(font.render("image_id", True, HEADER_COLOR), (MID_X, 24))
    surface.blit(font.render("description", True, HEADER_COLOR), (RIGHT_X, 24))


def main():
    parser = argparse.ArgumentParser(description="Vertical credits-style metadata scroller")
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parent / "metas.csv"),
        help="Path to headerless metas.csv",
    )
    parser.add_argument(
        "--rows-per-second",
        type=float,
        default=ROWS_PER_SECOND,
        help="Scroll speed in rows per second",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError("No rows found in CSV")

    pygame.init()
    pygame.display.set_caption("Scroller Test")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    font_row = pygame.font.SysFont("Menlo", 24)
    font_header = pygame.font.SysFont("Menlo", 20, bold=True)

    offset_y = float(SCREEN_HEIGHT)
    speed_px_s = ROW_HEIGHT * max(args.rows_per_second, 0.0)
    frozen = False

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not frozen:
            offset_y -= speed_px_s * dt
            last_row_y = offset_y + (len(rows) - 1) * ROW_HEIGHT
            if last_row_y <= TOP_MARGIN:
                offset_y = TOP_MARGIN - (len(rows) - 1) * ROW_HEIGHT
                frozen = True

        screen.fill(BG_COLOR)
        draw_headers(screen, font_header)

        for idx, row in enumerate(rows):
            y = int(offset_y + idx * ROW_HEIGHT)
            if y < TOP_MARGIN - ROW_HEIGHT or y > SCREEN_HEIGHT - BOTTOM_MARGIN:
                continue

            if idx % 2 == 0:
                pygame.draw.rect(screen, (12, 14, 18), (LEFT_X, y - 2, SCREEN_WIDTH - 56, ROW_HEIGHT - 4), 0)

            draw_text_clipped(screen, font_row, row.objects_text, TEXT_COLOR, LEFT_X, y + 8, LEFT_W)
            draw_text_clipped(screen, font_row, row.image_id, ID_COLOR, MID_X, y + 8, MID_W)
            draw_text_fade_right(
                screen,
                font_row,
                row.description,
                MUTED_TEXT,
                RIGHT_X,
                y + 8,
                RIGHT_W,
                fade_width=70,
                bg_color=BG_COLOR,
            )

            pygame.draw.line(screen, GRID_COLOR, (LEFT_X, y + ROW_HEIGHT - 1), (SCREEN_WIDTH - 28, y + ROW_HEIGHT - 1), 1)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
