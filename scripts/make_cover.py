"""
CropDoc Cover Image v2 — 마케팅 최고 퀄리티
전체 구도 개선: 왼쪽 임팩트 스토리 + 오른쪽 UI 패널
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math, os, random

W, H = 1600, 900
OUT = "/tmp/cropdoc_cover_v2.png"

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
FONT_REG  = os.path.join(FONT_DIR, "DejaVuSans.ttf")
FONT_MONO = os.path.join(FONT_DIR, "DejaVuSansMono-Bold.ttf")

def fnt(path, size):
    try: return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

def lerp(c1, c2, t):
    return tuple(int(c1[i]+(c2[i]-c1[i])*t) for i in range(3))

# ── 팔레트 ──────────────────────────────────────────────────
BG1    = (6,  18,  12)
BG2    = (4,  30,  20)
ACCENT = (0,  220, 120)    # 에메랄드
NEON   = (80, 255, 160)    # 네온그린
GOLD   = (255, 210, 60)
TEAL   = (0,  190, 170)
PURPLE = (170, 100, 255)
WHITE  = (255, 255, 255)
GREY   = (160, 185, 170)
RED    = (255, 80,  80)
ORANGE = (255, 150, 40)

# ══════════════════════════════════════════════════════════════
# 배경 레이어드 그라디언트
# ══════════════════════════════════════════════════════════════
base = Image.new("RGB", (W, H))
bd = ImageDraw.Draw(base)
for y in range(H):
    t = y/H
    # 수직 + 약간 방사형 느낌
    c = lerp(BG1, BG2, t)
    bd.line([(0,y),(W,y)], fill=c)
# 좌상단 미묘한 블루 틴트
for y in range(H//3):
    t = 1-(y/(H//3))
    bd.line([(0,y),(W//3,y)], fill=(
        min(255, base.getpixel((0,y))[0]+int(t*8)),
        base.getpixel((0,y))[1],
        min(255, base.getpixel((0,y))[2]+int(t*15))
    ))

img = base.copy()
draw = ImageDraw.Draw(img)

# ══════════════════════════════════════════════════════════════
# 육각형 그리드 배경 (전체 우측 절반)
# ══════════════════════════════════════════════════════════════
hex_l = Image.new("RGBA", (W, H), (0,0,0,0))
hd = ImageDraw.Draw(hex_l)
R = 44
for row in range(-1, int(H/(R*1.5))+3):
    for col in range(-1, int(W/(R*1.732))+3):
        cx = col * R*1.732 + (row%2)*R*0.866
        cy = row * R*1.5
        # 우측 영역만 강조
        rel_x = (cx - W*0.50) / (W*0.50)
        if rel_x < -0.5: 
            alpha = max(0, int(8*(1+rel_x*2)))
        else:
            dist = math.sqrt(max(0,(cx-W*0.75))**2 + (cy-H*0.5)**2)
            alpha = max(0, int(22 - dist*0.015))
        if alpha > 0:
            pts = [(cx+R*math.cos(math.radians(60*i-30)), cy+R*math.sin(math.radians(60*i-30))) for i in range(6)]
            hd.polygon(pts, outline=(ACCENT[0],ACCENT[1],ACCENT[2],alpha))

img = Image.alpha_composite(img.convert("RGBA"), hex_l).convert("RGB")
draw = ImageDraw.Draw(img)

# ══════════════════════════════════════════════════════════════
# 글로우 효과 (멀티 레이어)
# ══════════════════════════════════════════════════════════════
glow_l = Image.new("RGBA", (W, H), (0,0,0,0))
gd = ImageDraw.Draw(glow_l)

def add_glow(d, cx, cy, maxr, col, max_alpha=60):
    for r in range(maxr, 0, -15):
        a = int(max_alpha * (1-r/maxr)**1.8)
        d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(*col, a))

add_glow(gd, int(W*0.80), int(H*0.15), 280, ACCENT, 50)
add_glow(gd, int(W*0.10), int(H*0.88), 180, TEAL, 35)
add_glow(gd, int(W*0.48), int(H*0.50), 120, ACCENT, 15)

img = Image.alpha_composite(img.convert("RGBA"), glow_l).convert("RGB")
draw = ImageDraw.Draw(img)

# ══════════════════════════════════════════════════════════════
# 세로 구분선 (좌/우 분리)
# ══════════════════════════════════════════════════════════════
div_x = 840
div_l = Image.new("RGBA", (W, H), (0,0,0,0))
dd = ImageDraw.Draw(div_l)
for y in range(H):
    t = abs(y - H//2)/(H//2)
    a = int(60*(1-t**1.5))
    if a > 0:
        dd.point((div_x, y), fill=(*ACCENT, a))
        dd.point((div_x+1, y), fill=(*ACCENT, a//3))
img = Image.alpha_composite(img.convert("RGBA"), div_l).convert("RGB")
draw = ImageDraw.Draw(img)

# ══════════════════════════════════════════════════════════════
# 왼쪽 영역 — 임팩트 스토리
# ══════════════════════════════════════════════════════════════

# 최상단 배지
bx, by = 52, 50
badge_l = Image.new("RGBA", (W, H), (0,0,0,0))
bld = ImageDraw.Draw(badge_l)
bld.rounded_rectangle([bx, by, bx+360, by+34], radius=17,
    fill=(0,100,50,230), outline=(*ACCENT, 200), width=2)
img = Image.alpha_composite(img.convert("RGBA"), badge_l).convert("RGB")
draw = ImageDraw.Draw(img)
draw.text((bx+52, by+8), "GEMMA 4 GOOD HACKATHON  ·  2026", 
          font=fnt(FONT_BOLD, 14), fill=NEON)
# 트로피 아이콘 영역
draw.rounded_rectangle([bx+6, by+6, bx+42, by+28], radius=8, fill=(0,150,70))
draw.text((bx+12, by+7), "🏆", font=fnt(FONT_REG, 16), fill=WHITE)

# ────── 메인 타이틀 ──────────────────────────────────────────
ty = 108

# CropDoc — 초대형 (그라디언트 느낌은 레이어로)
title_f = fnt(FONT_BOLD, 96)
# 그림자 레이어
for dx, dy, alpha in [(4,5,40),(2,3,60),(1,2,80)]:
    shadow_l = Image.new("RGBA", (W, H), (0,0,0,0))
    sd = ImageDraw.Draw(shadow_l)
    sd.text((52+dx, ty+dy), "CropDoc", font=title_f, fill=(0,40,20,alpha))
    img = Image.alpha_composite(img.convert("RGBA"), shadow_l).convert("RGB")

draw = ImageDraw.Draw(img)
draw.text((52, ty), "CropDoc", font=title_f, fill=NEON)

# 서브제목
draw.text((54, ty+102), "Plant Disease AI", font=fnt(FONT_BOLD, 38), fill=WHITE)
draw.text((56, ty+148), "Offline · Multimodal · 10 Languages", 
          font=fnt(FONT_REG, 20), fill=GREY)

# ────── 임팩트 구분선 ─────────────────────────────────────────
line_y = ty + 185
draw.line([(52, line_y), (780, line_y)], fill=(40,100,60), width=1)

# ────── 3가지 핵심 수치 카드 ─────────────────────────────────
card_y = line_y + 20
card_data = [
    ("500M", "Smallholder\nFarmers", NEON, "🌱"),
    ("72",   "Disease\nClasses",   TEAL,  "🔬"),
    ("99.3%","Diagnosis\nAccuracy", GOLD,  "✅"),
    ("$220B","Annual Loss\nPrevented", ORANGE, "💰"),
]
cw, ch = 183, 115
cx = 52
for val, label, col, icon in card_data:
    # 카드 배경
    card_l = Image.new("RGBA", (W, H), (0,0,0,0))
    cd = ImageDraw.Draw(card_l)
    cd.rounded_rectangle([cx, card_y, cx+cw, card_y+ch], radius=14,
        fill=(col[0]//7, col[1]//7, col[2]//7, 210),
        outline=(*col, 130), width=1)
    # 상단 엑센트 라인
    cd.rounded_rectangle([cx+1, card_y+1, cx+cw-1, card_y+5], radius=3, fill=(*col, 180))
    img = Image.alpha_composite(img.convert("RGBA"), card_l).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    draw.text((cx+12, card_y+12), val, font=fnt(FONT_BOLD, 34), fill=col)
    for li, ll in enumerate(label.split('\n')):
        draw.text((cx+12, card_y+56+li*19), ll, font=fnt(FONT_REG, 14), fill=GREY)
    cx += cw + 12

# ────── 기술 스택 (가로 필 형태) ────────────────────────────
stack_y = card_y + ch + 20
draw.text((52, stack_y), "POWERED BY", font=fnt(FONT_MONO, 11), fill=(80,140,100))
stack_y += 18

stack_items = [
    ("Gemma 4 E4B-IT", PURPLE),
    ("EfficientNetV2-S", ACCENT),
    ("SwinV2-S", TEAL),
    ("ConvNeXt-Base", TEAL),
    ("PEFT LoRA", GOLD),
    ("BitsAndBytes NF4", ORANGE),
]
sx = 52
for name, col in stack_items:
    text_w = len(name)*8 + 22
    pill_l = Image.new("RGBA", (W, H), (0,0,0,0))
    pd = ImageDraw.Draw(pill_l)
    pd.rounded_rectangle([sx, stack_y, sx+text_w, stack_y+26], radius=13,
        fill=(col[0]//5, col[1]//5, col[2]//5, 200),
        outline=(*col, 160), width=1)
    img = Image.alpha_composite(img.convert("RGBA"), pill_l).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((sx+10, stack_y+6), name, font=fnt(FONT_REG, 13), fill=col)
    sx += text_w + 8

# ────── 트랙 뱃지 ──────────────────────────────────────────
track_y = stack_y + 40
draw.text((52, track_y), "COMPETING IN", font=fnt(FONT_MONO, 11), fill=(80,140,100))
track_y += 18

tracks = [
    ("🌍  Global Resilience", (0,160,80)),
    ("🌵  Cactus Prize",      (190,110,0)),
    ("📡  Edge AI Special",   (0,120,200)),
]
tx2 = 52
for tname, tcol in tracks:
    tw = len(tname)*8 + 22
    tl = Image.new("RGBA", (W, H), (0,0,0,0))
    td = ImageDraw.Draw(tl)
    td.rounded_rectangle([tx2, track_y, tx2+tw, track_y+30], radius=15,
        fill=(tcol[0]//4, tcol[1]//4, tcol[2]//4, 220),
        outline=(*tcol, 200), width=2)
    img = Image.alpha_composite(img.convert("RGBA"), tl).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((tx2+10, track_y+7), tname, font=fnt(FONT_BOLD, 14), fill=WHITE)
    tx2 += tw + 12

# ══════════════════════════════════════════════════════════════
# 오른쪽 영역 — CropDoc UI 패널
# ══════════════════════════════════════════════════════════════
px, py = 860, 55
pw, ph_panel = 700, 795

# 패널 배경 (글래스모피즘)
panel_l = Image.new("RGBA", (W, H), (0,0,0,0))
pld = ImageDraw.Draw(panel_l)

# 외부 글로우
for i in range(15, 0, -1):
    a = int(25*(1-i/15)**2)
    pld.rounded_rectangle([px-i, py-i, px+pw+i, py+ph_panel+i],
        radius=20, fill=(0,200,100,a))

# 메인 패널
pld.rounded_rectangle([px, py, px+pw, py+ph_panel], radius=18,
    fill=(10,32,20,225),
    outline=(*ACCENT, 100), width=2)

img = Image.alpha_composite(img.convert("RGBA"), panel_l).convert("RGB")
draw = ImageDraw.Draw(img)

# 패널 헤더
header_l = Image.new("RGBA", (W, H), (0,0,0,0))
hl = ImageDraw.Draw(header_l)
hl.rounded_rectangle([px, py, px+pw, py+54], radius=18,
    fill=(0,140,70,200))
hl.rectangle([px, py+38, px+pw, py+54], fill=(0,140,70,200))
img = Image.alpha_composite(img.convert("RGBA"), header_l).convert("RGB")
draw = ImageDraw.Draw(img)

# 맥OS 닷
for xi, col in [(px+16, RED), (px+34, ORANGE), (px+52, (50,210,80))]:
    draw.ellipse([xi-6,py+21,xi+6,py+33], fill=col)

draw.text((px+70, py+16), "CropDoc AI  —  Plant Disease Diagnosis Engine  v2.1",
          font=fnt(FONT_MONO, 13), fill=(200,255,220))

# ──── 패널 내부 ─────────────────────────────────────────────
iy = py + 68

# [A] 이미지 스캔 영역 (왼쪽)
ia_x, ia_y, ia_w, ia_h = px+18, iy, 310, 250

scan_l = Image.new("RGBA", (W, H), (0,0,0,0))
sl = ImageDraw.Draw(scan_l)
sl.rounded_rectangle([ia_x, ia_y, ia_x+ia_w, ia_y+ia_h], radius=10,
    fill=(8,45,25,220), outline=(*ACCENT, 50), width=1)
img = Image.alpha_composite(img.convert("RGBA"), scan_l).convert("RGB")
draw = ImageDraw.Draw(img)

# 잎사귀 시각화
lx, ly = ia_x + ia_w//2, ia_y + ia_h//2 - 10
draw.ellipse([lx-65, ly-80, lx+65, ly+65], fill=(14,90,40),
             outline=(0,200,80,150), width=2)
draw.line([(lx, ly-75), (lx, ly+60)], fill=(0,200,80), width=2)
for ang, ofs in [(-30,0.6), (30,0.6), (-20,0.2), (20,0.2)]:
    a = math.radians(ang)
    ex = lx + 55*math.cos(a)
    ey = ly + 55*math.sin(a)*0.5 + (ly-75)*(1-ofs)
    mid_x = (lx+ex)/2
    mid_y = (ly + ey)/2 - 10
    draw.line([(lx, ly+(ly-75)*(1-ofs)), (int(ex), int(ey))], fill=(0,180,70), width=1)

random.seed(42)
for _ in range(12):
    bx2 = lx + random.randint(-50, 50)
    by2 = ly + random.randint(-60, 40)
    br2 = random.randint(4, 12)
    draw.ellipse([bx2-br2,by2-br2,bx2+br2,by2+br2], fill=(140,60,15))

# 스캔 라인 (에니메이션 효과 — 정지 프레임)
for si in range(0, ia_h, 3):
    a = int(12 + 8*math.sin(si*0.25))
    scan_y = ia_y + si
    if scan_y <= ia_y + ia_h:
        draw.line([(ia_x+1, scan_y),(ia_x+ia_w-1, scan_y)], fill=(0,255,100,a))

# 스캔 상태 표시
draw.text((ia_x+8, ia_y+8), "📷  ANALYZING IMAGE", font=fnt(FONT_MONO, 12), fill=NEON)

scan_bar_y = ia_y + ia_h - 24
draw.rectangle([ia_x+8, scan_bar_y, ia_x+ia_w-8, scan_bar_y+10],
               fill=(20,55,30), outline=(ACCENT[0],ACCENT[1],ACCENT[2]))
draw.rectangle([ia_x+8, scan_bar_y, ia_x+8+int((ia_w-16)*0.87), scan_bar_y+10], fill=ACCENT)
draw.text((ia_x+ia_w-50, scan_bar_y), "87%", font=fnt(FONT_BOLD, 12), fill=NEON)

# [B] 진단 결과 (오른쪽)
ra_x = px + 345
ra_y = iy

draw.text((ra_x, ra_y), "DIAGNOSIS RESULT", font=fnt(FONT_MONO, 12), fill=(ACCENT[0],ACCENT[1],ACCENT[2]))
ra_y += 22

draw.text((ra_x, ra_y), "Tomato Late Blight", font=fnt(FONT_BOLD, 26), fill=WHITE)
ra_y += 34

# 신뢰도
draw.text((ra_x, ra_y), "Confidence Score", font=fnt(FONT_REG, 13), fill=GREY)
ra_y += 17
conf = 0.9933
bar_w2 = 330
draw.rectangle([ra_x, ra_y, ra_x+bar_w2, ra_y+14], fill=(18,48,28), outline=(55,110,65))
draw.rectangle([ra_x, ra_y, ra_x+int(bar_w2*conf), ra_y+14], fill=ACCENT)
draw.text((ra_x+bar_w2+8, ra_y), "99.3%", font=fnt(FONT_BOLD, 14), fill=NEON)
ra_y += 26

# 심각도 + 모델
sev_l = Image.new("RGBA", (W, H), (0,0,0,0))
sevd = ImageDraw.Draw(sev_l)
sevd.rounded_rectangle([ra_x, ra_y, ra_x+120, ra_y+28], radius=14, fill=(80,45,0,200), outline=(*GOLD, 200), width=2)
img = Image.alpha_composite(img.convert("RGBA"), sev_l).convert("RGB")
draw = ImageDraw.Draw(img)
draw.text((ra_x+12, ra_y+7), "⚠  MODERATE", font=fnt(FONT_BOLD, 13), fill=GOLD)

sev_l2 = Image.new("RGBA", (W, H), (0,0,0,0))
s2d = ImageDraw.Draw(sev_l2)
s2d.rounded_rectangle([ra_x+130, ra_y, ra_x+290, ra_y+28], radius=14, fill=(40,0,80,200), outline=(*PURPLE, 180), width=1)
img = Image.alpha_composite(img.convert("RGBA"), sev_l2).convert("RGB")
draw = ImageDraw.Draw(img)
draw.text((ra_x+142, ra_y+7), "🤖 Gemma 4 E4B", font=fnt(FONT_BOLD, 13), fill=PURPLE)
ra_y += 44

# 구분선
draw.line([(ra_x, ra_y),(ra_x+bar_w2+30, ra_y)], fill=(40,90,55), width=1)
ra_y += 14

draw.text((ra_x, ra_y), "TREATMENT PLAN", font=fnt(FONT_MONO, 12), fill=TEAL)
ra_y += 20

tx_items = [
    ("①", "Apply copper-based fungicide within 48h"),
    ("②", "Remove and destroy infected leaves immediately"),
    ("③", "Avoid overhead irrigation; use drip system"),
]
for num, tx in tx_items:
    draw.text((ra_x, ra_y), num, font=fnt(FONT_BOLD, 14), fill=ACCENT)
    draw.text((ra_x+22, ra_y), tx, font=fnt(FONT_REG, 14), fill=(200,235,210))
    ra_y += 20

ra_y += 8
draw.line([(ra_x, ra_y),(ra_x+bar_w2+30, ra_y)], fill=(40,90,55), width=1)
ra_y += 14

draw.text((ra_x, ra_y), "LANGUAGE / 언어 선택", font=fnt(FONT_MONO, 12), fill=TEAL)
ra_y += 20

lang_items = [("🇺🇸","EN"), ("🇰🇷","KO"), ("🇪🇸","ES"), ("🇧🇷","PT"), ("🇨🇳","ZH"), ("🇮🇳","HI"), ("🇫🇷","FR"), ("🇧🇩","BN"), ("🇸🇦","AR"), ("🇷🇺","RU")]
lx3 = ra_x
for flag, code in lang_items:
    active = (code == "KO")
    pill_l3 = Image.new("RGBA", (W, H), (0,0,0,0))
    p3d = ImageDraw.Draw(pill_l3)
    p3d.rounded_rectangle([lx3, ra_y, lx3+46, ra_y+24], radius=12,
        fill=(0,100,55,220) if active else (15,50,28,180),
        outline=(*NEON, 220) if active else (*ACCENT, 60), width=1)
    img = Image.alpha_composite(img.convert("RGBA"), pill_l3).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((lx3+4, ra_y+5), f"{flag}{code}", font=fnt(FONT_REG, 12),
              fill=NEON if active else GREY)
    lx3 += 52
    if lx3 > ra_x+bar_w2:
        lx3 = ra_x
        ra_y += 28

ra_y += 30
draw.line([(ra_x, ra_y),(ra_x+bar_w2+30, ra_y)], fill=(40,90,55), width=1)
ra_y += 14

# 오프라인 강조
offline_l = Image.new("RGBA", (W, H), (0,0,0,0))
old = ImageDraw.Draw(offline_l)
old.rounded_rectangle([ra_x, ra_y, ra_x+165, ra_y+30], radius=15,
    fill=(0,50,90,210), outline=(0,160,230,200), width=2)
img = Image.alpha_composite(img.convert("RGBA"), offline_l).convert("RGB")
draw = ImageDraw.Draw(img)
draw.text((ra_x+10, ra_y+8), "📡  OFFLINE-FIRST AI", font=fnt(FONT_BOLD, 13), fill=(80,210,255))

edge_l = Image.new("RGBA", (W, H), (0,0,0,0))
eld = ImageDraw.Draw(edge_l)
eld.rounded_rectangle([ra_x+175, ra_y, ra_x+360, ra_y+30], radius=15,
    fill=(40,0,70,200), outline=(*PURPLE, 180), width=1)
img = Image.alpha_composite(img.convert("RGBA"), edge_l).convert("RGB")
draw = ImageDraw.Draw(img)
draw.text((ra_x+187, ra_y+8), "⚡  Edge Deploy Ready", font=fnt(FONT_BOLD, 13), fill=PURPLE)

# ──── 하단 통계 바 ──────────────────────────────────────────
iy = py + 340
draw.line([(px+12, iy),(px+pw-12, iy)], fill=(35,85,50), width=1)
iy += 20

stats = [
    ("72", "Crop Disease Classes", ACCENT),
    ("10", "Languages Supported", TEAL),
    ("99.3%", "Test Accuracy", GOLD),
    ("2,148", "Fine-tune Samples", PURPLE),
    ("141MB", "LoRA Adapter Size", ORANGE),
]
sw3 = (pw-36) // len(stats)
for si, (val, lbl, col) in enumerate(stats):
    sx2 = px+18 + si*sw3
    draw.text((sx2, iy), val, font=fnt(FONT_BOLD, 28), fill=col)
    draw.text((sx2, iy+32), lbl, font=fnt(FONT_REG, 12), fill=GREY)

iy += 68
draw.line([(px+12, iy),(px+pw-12, iy)], fill=(35,85,50), width=1)
iy += 18

# ──── 멀티언어 예시 출력 ──────────────────────────────────────
draw.text((px+18, iy), "MULTILINGUAL OUTPUT EXAMPLE", font=fnt(FONT_MONO, 11), fill=(70,140,90))
iy += 18

sample_l = Image.new("RGBA", (W, H), (0,0,0,0))
sampd = ImageDraw.Draw(sample_l)
sampd.rounded_rectangle([px+18, iy, px+pw-18, iy+180], radius=8,
    fill=(5,30,16,230), outline=(*ACCENT, 30), width=1)
img = Image.alpha_composite(img.convert("RGBA"), sample_l).convert("RGB")
draw = ImageDraw.Draw(img)

samples = [
    ("🇺🇸 EN:", "Diagnosis: Tomato Late Blight | Severity: Moderate", WHITE),
    ("🇰🇷 KO:", "진단: Tomato Late Blight | 심각도: 보통", (180,255,200)),
    ("🇪🇸 ES:", "Diagnóstico: Tomato Late Blight | Gravedad: Moderada", (180,220,255)),
    ("🇨🇳 ZH:", "诊断: Tomato Late Blight | 严重性: 中等", (255,220,180)),
    ("🇮🇳 HI:", "निदान: Tomato Late Blight | गंभीरता: मध्यम", (220,180,255)),
]
for fi, (flag, text, col) in enumerate(samples):
    draw.text((px+28, iy+8+fi*30), flag, font=fnt(FONT_BOLD, 13), fill=ACCENT)
    draw.text((px+90, iy+8+fi*30), text, font=fnt(FONT_REG, 13), fill=col)

iy += 195

# ──── 아키텍처 다이어그램 (미니) ─────────────────────────────
draw.text((px+18, iy), "HYBRID ARCHITECTURE", font=fnt(FONT_MONO, 11), fill=(70,140,90))
iy += 18

arch_items = [
    ("📸 Image Input", GREY, 40),
    ("→", WHITE, 0),
    ("🧠 CNN Ensemble\n(EfficientNet+Swin+ConvNeXt)", ACCENT, 40),
    ("→", WHITE, 0),
    ("conf≥0.90?", GOLD, 30),
    ("→", WHITE, 0),
    ("🤖 Gemma 4 E4B\n(FOCUSED verify)", PURPLE, 40),
    ("→", WHITE, 0),
    ("✅ Diagnosis +\nReport (10 langs)", NEON, 40),
]
ax = px+18
ay_arch = iy
arch_l = Image.new("RGBA", (W, H), (0,0,0,0))
ad = ImageDraw.Draw(arch_l)

for atext, acol, abx in arch_items:
    if atext == "→":
        draw.text((ax, ay_arch+14), "→", font=fnt(FONT_BOLD, 22), fill=acol)
        ax += 22
    else:
        lines_a = atext.split('\n')
        text_w = max(len(l)*7 for l in lines_a) + 16
        bh = len(lines_a)*18 + 14
        ad.rounded_rectangle([ax, ay_arch, ax+text_w, ay_arch+bh], radius=7,
            fill=(acol[0]//5, acol[1]//5, acol[2]//5, 200),
            outline=(*acol, 150), width=1)
        img2 = Image.alpha_composite(img.convert("RGBA"), arch_l).convert("RGB")
        img = img2
        draw = ImageDraw.Draw(img)
        arch_l = Image.new("RGBA", (W, H), (0,0,0,0))
        ad = ImageDraw.Draw(arch_l)
        for li2, ll2 in enumerate(lines_a):
            draw.text((ax+8, ay_arch+7+li2*18), ll2, font=fnt(FONT_REG, 12), fill=acol)
        ax += text_w + 4

img = Image.alpha_composite(img.convert("RGBA"), arch_l).convert("RGB")
draw = ImageDraw.Draw(img)

# ══════════════════════════════════════════════════════════════
# 하단 푸터
# ══════════════════════════════════════════════════════════════
footer_y = H - 52
footer_l = Image.new("RGBA", (W, H), (0,0,0,0))
fd = ImageDraw.Draw(footer_l)
fd.rectangle([0, footer_y, W, H], fill=(4,16,10,240))
fd.line([(0, footer_y),(W, footer_y)], fill=(*ACCENT, 60), width=1)
img = Image.alpha_composite(img.convert("RGBA"), footer_l).convert("RGB")
draw = ImageDraw.Draw(img)

draw.text((40, footer_y+15), 
    "🤗 huggingface.co/spaces/noivan/cropdoc    ·    💻 github.com/noivan0/cropdoc    ·    🤖 Powered by Gemma 4 E4B  (CC-BY 4.0)",
    font=fnt(FONT_REG, 13), fill=(80,160,110))

# ══════════════════════════════════════════════════════════════
# 비네팅 + 최종 샤프닝
# ══════════════════════════════════════════════════════════════
vig_l = Image.new("RGBA", (W, H), (0,0,0,0))
vigd = ImageDraw.Draw(vig_l)
for ri in range(max(W,H)//2, 0, -10):
    t = ri/(max(W,H)//2)
    a = int(55*(1-t)**2.5)
    vigd.ellipse([W//2-ri, H//2-ri, W//2+ri, H//2+ri], fill=(0,0,0,a))
img = Image.alpha_composite(img.convert("RGBA"), vig_l).convert("RGB")

img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=3))
img.save(OUT, "PNG", optimize=False)
print(f"✅ Cover Image v2 저장: {OUT}  ({img.size})")
