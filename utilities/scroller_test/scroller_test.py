
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

COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

ALL_CLASSES = {**COCO_CLASSES, **CLASSES}

COLOR_MODES = {
    0: "centerwhitehorizontal",
    1: "centerallwhite",
    2: "alternatinghorizontal",
    3: "alternatinghorizontal_flipflop"
}
USING_MODE = 3

SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 1920
ROWS_PER_SECOND = 9.0
ROW_HEIGHT = 54
TOP_MARGIN = 28
BOTTOM_MARGIN = 64

BG_COLOR = (8, 9, 12)
GRID_COLOR = (32, 35, 42)
# TEXT_COLOR = (228, 233, 242)
TEXT_COLOR = (228, 233, 242)
MUTED_TEXT = (192, 200, 215)
# MUTED_TEXT = (136, 146, 162)
ID_TEXT_COLOR = (24, 27, 32)
CENTER_BG_COLOR = (245, 245, 245)
ZEBRA_DARK_A = (8, 10, 14)
ZEBRA_DARK_B = (30, 34, 45)
# ZEBRA_DARK_B = (26, 30, 38)
# ZEBRA_LIGHT_A = (248, 248, 248)
ZEBRA_LIGHT_A = (237, 242, 245)
ZEBRA_LIGHT_B = (210, 213, 220)
# ZEBRA_LIGHT_B = (220, 224, 230)

OUTER_PAD = 28
GUTTER = 16
MID_W = 150
MID_X = (SCREEN_WIDTH - MID_W) // 2
SIDE_W = (SCREEN_WIDTH - (OUTER_PAD * 2) - MID_W - (GUTTER * 2)) // 2
LEFT_X = OUTER_PAD
LEFT_W = SIDE_W
RIGHT_X = MID_X + MID_W + GUTTER
RIGHT_W = SIDE_W
MID_TEXT_PAD = 14


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
    return ALL_CLASSES.get(class_id, f"class_{class_id}")


def render_detection_text(detections: List[dict]) -> str:
    if not detections:
        return "none"
    show_conf = len(detections) == 1
    parts = []
    for det in detections:
        class_id = det.get("class_id")
        try:
            class_id = int(class_id)
        except (TypeError, ValueError):
            continue
        label = f"class_id:{class_id} ({class_label(class_id)})"
        if show_conf:
            conf = det.get("conf")
            try:
                conf_text = f"{float(conf):.2f}"
            except (TypeError, ValueError):
                conf_text = "n/a"
            label = f"{label} conf:{conf_text}"
        parts.append(label)
    return ", ".join(parts) if parts else "none"


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


def draw_text_clipped_right(surface, font, text, color, x, y, width):
    text_surface = font.render(text, True, color)
    text_w = text_surface.get_width()
    text_h = text_surface.get_height()

    if text_w <= width:
        surface.blit(text_surface, (x + width - text_w, y))
        return

    clip_surface = pygame.Surface((width, text_h), pygame.SRCALPHA)
    clip_surface.blit(text_surface, (width - text_w, 0))
    surface.blit(clip_surface, (x, y))


def draw_text_fade_right(
    surface,
    font,
    text,
    color,
    x,
    y,
    width,
    fade_width,
    fade_top,
    fade_height,
    fade_color,
):
    # Draw text clipped to column width first.
    draw_text_clipped(surface, font, text, color, x, y, width)

    # Fade across the full row height so zebra stripes and text fade together.
    fade_surface = pygame.Surface((fade_width, fade_height), pygame.SRCALPHA)
    for i in range(fade_width):
        alpha = int(255 * (i / max(1, fade_width - 1)))
        pygame.draw.line(
            fade_surface,
            (fade_color[0], fade_color[1], fade_color[2], alpha),
            (i, 0),
            (i, fade_height),
        )
    surface.blit(fade_surface, (x + width - fade_width, fade_top))


def draw_text_fade_left(
    surface,
    font,
    text,
    color,
    x,
    y,
    width,
    fade_width,
    fade_top,
    fade_height,
    fade_color,
):
    # Draw right-aligned text clipped to column width first.
    draw_text_clipped_right(surface, font, text, color, x, y, width)

    # Fade across the full row height so zebra stripes and text fade together.
    fade_surface = pygame.Surface((fade_width, fade_height), pygame.SRCALPHA)
    for i in range(fade_width):
        alpha = int(255 * (1.0 - (i / max(1, fade_width - 1))))
        pygame.draw.line(
            fade_surface,
            (fade_color[0], fade_color[1], fade_color[2], alpha),
            (i, 0),
            (i, fade_height),
        )
    surface.blit(fade_surface, (x, fade_top))


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
    offset_y = float(SCREEN_HEIGHT)
    speed_px_s = ROW_HEIGHT * max(args.rows_per_second, 0.0)
    frozen = False
    color_mode = COLOR_MODES.get(USING_MODE, COLOR_MODES[0])

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
        if color_mode in {"centerwhitehorizontal", "centerallwhite"}:
            pygame.draw.rect(screen, CENTER_BG_COLOR, (MID_X, 0, MID_W, SCREEN_HEIGHT), 0)

        for idx, row in enumerate(rows):
            y = int(offset_y + idx * ROW_HEIGHT)
            if y < TOP_MARGIN - ROW_HEIGHT or y > SCREEN_HEIGHT - BOTTOM_MARGIN:
                continue

            image_id_color = ID_TEXT_COLOR

            if color_mode == "alternatinghorizontal_flipflop":
                side_row_color = ZEBRA_DARK_A if idx % 2 == 0 else ZEBRA_DARK_B
                center_row_color = ZEBRA_DARK_B if idx % 2 == 0 else ZEBRA_DARK_A
                pygame.draw.rect(screen, side_row_color, (LEFT_X, y, LEFT_W, ROW_HEIGHT), 0)
                pygame.draw.rect(screen, center_row_color, (MID_X, y, MID_W, ROW_HEIGHT), 0)
                pygame.draw.rect(screen, side_row_color, (RIGHT_X, y, RIGHT_W, ROW_HEIGHT), 0)
                image_id_color = TEXT_COLOR
            elif color_mode == "alternatinghorizontal":
                dark_row_color = ZEBRA_DARK_A if idx % 2 == 0 else ZEBRA_DARK_B
                light_row_color = ZEBRA_LIGHT_A if idx % 2 == 0 else ZEBRA_LIGHT_B
                pygame.draw.rect(screen, dark_row_color, (LEFT_X, y, LEFT_W, ROW_HEIGHT), 0)
                pygame.draw.rect(screen, light_row_color, (MID_X, y, MID_W, ROW_HEIGHT), 0)
                pygame.draw.rect(screen, dark_row_color, (RIGHT_X, y, RIGHT_W, ROW_HEIGHT), 0)
            elif idx % 2 == 0:
                if color_mode != "alternatinghorizontal":
                    pygame.draw.rect(screen, (12, 14, 18), (LEFT_X, y - 2, LEFT_W, ROW_HEIGHT - 4), 0)
                    pygame.draw.rect(screen, (12, 14, 18), (RIGHT_X, y - 2, RIGHT_W, ROW_HEIGHT - 4), 0)

            draw_text_fade_left(
                screen,
                font_row,
                row.objects_text,
                MUTED_TEXT,
                LEFT_X,
                y + 8,
                LEFT_W,
                fade_width=70,
                fade_top=y,
                fade_height=ROW_HEIGHT,
                fade_color=BG_COLOR,
            )
            draw_text_clipped(
                screen,
                font_row,
                row.image_id,
                image_id_color,
                MID_X + MID_TEXT_PAD,
                y + 8,
                MID_W - (MID_TEXT_PAD * 2),
            )
            draw_text_fade_right(
                screen,
                font_row,
                row.description,
                MUTED_TEXT,
                RIGHT_X,
                y + 8,
                RIGHT_W,
                fade_width=70,
                fade_top=y,
                fade_height=ROW_HEIGHT,
                fade_color=BG_COLOR,
            )

            if color_mode in {"alternatinghorizontal", "alternatinghorizontal_flipflop"}:
                pass
            elif color_mode == "centerallwhite":
                pygame.draw.line(screen, GRID_COLOR, (LEFT_X, y + ROW_HEIGHT - 1), (LEFT_X + LEFT_W, y + ROW_HEIGHT - 1), 1)
                pygame.draw.line(screen, GRID_COLOR, (RIGHT_X, y + ROW_HEIGHT - 1), (SCREEN_WIDTH - OUTER_PAD, y + ROW_HEIGHT - 1), 1)
            else:
                pygame.draw.line(screen, GRID_COLOR, (LEFT_X, y + ROW_HEIGHT - 1), (SCREEN_WIDTH - OUTER_PAD, y + ROW_HEIGHT - 1), 1)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
