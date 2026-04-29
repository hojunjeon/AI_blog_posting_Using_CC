from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

W, H = 1200, 630

# --- 대각선 그라디언트 배경 (#1a73e8 -> #4285f4 -> #34a853) ---
def make_gradient():
    c1 = np.array([26, 115, 232], dtype=float)
    c2 = np.array([66, 133, 244], dtype=float)
    c3 = np.array([52, 168, 83],  dtype=float)
    x = np.arange(W, dtype=float)
    y = np.arange(H, dtype=float)
    xx, yy = np.meshgrid(x, y)
    t = (xx + yy) / (W + H - 2)
    mask = t < 0.5
    t1 = np.where(mask, t / 0.5, 0.0)
    t2 = np.where(~mask, (t - 0.5) / 0.5, 0.0)
    arr = np.zeros((H, W, 3), dtype=np.float32)
    for ch in range(3):
        arr[:, :, ch] = (
            np.where(mask,  c1[ch] + (c2[ch] - c1[ch]) * t1, 0) +
            np.where(~mask, c2[ch] + (c3[ch] - c2[ch]) * t2, 0)
        )
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

img = make_gradient().convert("RGBA")

# --- Phase 1: 배경 장식 도형 (흰색 10%) ---
ov1 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d1 = ImageDraw.Draw(ov1)
d1.rectangle([(50, 50), (150, 150)],            outline=(255, 255, 255, 25), width=3)
d1.ellipse(  [(W-160, 100), (W-80, 180)],       outline=(255, 255, 255, 25), width=3)
d1.rounded_rectangle([(150, H-160), (270, H-100)], radius=5, outline=(255, 255, 255, 25), width=3)
d1.rectangle([(W-160, H-210), (W-100, H-150)],  fill=(255, 255, 255, 25))
img = Image.alpha_composite(img, ov1)

# --- Phase 2: 반투명 UI 요소 ---
font_dir = "C:/Windows/Fonts"
def load_font(size, bold=False):
    candidates = ["malgunbd.ttf", "malgun.ttf"] if bold else ["malgun.ttf", "malgunbd.ttf"]
    for name in candidates:
        path = os.path.join(font_dir, name)
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

font_icon     = load_font(32, bold=True)
font_series   = load_font(22)
font_day      = load_font(28, bold=True)
font_title    = load_font(52, bold=True)
font_subtitle = load_font(36)
font_tag      = load_font(18)

BOX, BGAP = 80, 20
PAD_X, PAD_Y, TGAP = 20, 8, 15
icons = ["EDA", "SK", "전처리"]
tags  = ["EDA", "Scikit-learn", "표준화", "데이터전처리"]

_tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
def tsize(text, font):
    bb = _tmp.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]

icon_h    = BOX
series_h  = tsize("AI Learning Series", font_series)[1]
day_h     = tsize("Day 3", font_day)[1]
title_h   = tsize("EDA & Scikit-learn", font_title)[1]
sub_h     = tsize("데이터 탐색부터 ML 파이프라인까지", font_subtitle)[1]
tag_h     = tsize(tags[0], font_tag)[1] + PAD_Y * 2

total_h = icon_h + 30 + series_h + 15 + day_h + 12 + title_h + 15 + sub_h + 40 + tag_h
y = (H - total_h) // 2

icon_total_w = len(icons) * BOX + (len(icons) - 1) * BGAP
icon_x = (W - icon_total_w) // 2
icon_y = y

tag_widths  = [tsize(t, font_tag)[0] + PAD_X * 2 for t in tags]
total_tag_w = sum(tag_widths) + TGAP * (len(tags) - 1)
tag_x = (W - total_tag_w) // 2
tag_y = y + icon_h + 30 + series_h + 15 + day_h + 12 + title_h + 15 + sub_h + 40

ov2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d2 = ImageDraw.Draw(ov2)

for i in range(len(icons)):
    bx = icon_x + i * (BOX + BGAP)
    d2.rounded_rectangle(
        [(bx, icon_y), (bx + BOX, icon_y + BOX)],
        radius=15,
        fill=(255, 255, 255, 51),
        outline=(255, 255, 255, 128), width=3
    )

tx = tag_x
for tw in tag_widths:
    d2.rounded_rectangle(
        [(tx, tag_y), (tx + tw, tag_y + tag_h)],
        radius=tag_h // 2,
        fill=(255, 255, 255, 51),
        outline=(255, 255, 255, 77), width=1
    )
    tx += tw + TGAP

img = Image.alpha_composite(img, ov2)

# --- Phase 3: RGB 변환 후 텍스트 드로잉 ---
img = img.convert("RGB")
draw = ImageDraw.Draw(img)
WHITE = (255, 255, 255)

for i, label in enumerate(icons):
    bx = icon_x + i * (BOX + BGAP)
    bb = draw.textbbox((0, 0), label, font=font_icon)
    tw2, th2 = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(
        (bx + (BOX - tw2) // 2 - bb[0], icon_y + (BOX - th2) // 2 - bb[1]),
        label, font=font_icon, fill=WHITE
    )

cy = icon_y + BOX + 30

series = "AI Learning Series"
bb = draw.textbbox((0, 0), series, font=font_series)
draw.text(((W - (bb[2] - bb[0])) // 2 - bb[0], cy - bb[1]), series, font=font_series, fill=(230, 230, 230))
cy += series_h + 15

day_txt = "Day 3"
bb = draw.textbbox((0, 0), day_txt, font=font_day)
draw.text(((W - (bb[2] - bb[0])) // 2 - bb[0], cy - bb[1]), day_txt, font=font_day, fill=WHITE)
cy += day_h + 12

title = "EDA & Scikit-learn"
bb = draw.textbbox((0, 0), title, font=font_title)
tx2 = (W - (bb[2] - bb[0])) // 2 - bb[0]
draw.text((tx2 + 2, cy - bb[1] + 2), title, font=font_title, fill=(0, 0, 0))
draw.text((tx2, cy - bb[1]), title, font=font_title, fill=WHITE)
cy += title_h + 15

subtitle = "데이터 탐색부터 ML 파이프라인까지"
bb = draw.textbbox((0, 0), subtitle, font=font_subtitle)
draw.text(((W - (bb[2] - bb[0])) // 2 - bb[0], cy - bb[1]), subtitle, font=font_subtitle, fill=(240, 240, 240))
cy += sub_h + 40

tx = tag_x
for tag, tw in zip(tags, tag_widths):
    bb = draw.textbbox((0, 0), tag, font=font_tag)
    text_w = bb[2] - bb[0]
    draw.text(
        (tx + (tw - text_w) // 2 - bb[0], tag_y + PAD_Y - bb[1]),
        tag, font=font_tag, fill=WHITE
    )
    tx += tw + TGAP

# --- 저장 ---
out_path = os.path.join(os.path.dirname(__file__), "day3_thumbnail.png")
img.save(out_path)
print(f"Saved: {out_path}")
