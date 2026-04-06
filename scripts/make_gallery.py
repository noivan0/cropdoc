"""
CropDoc Media Gallery Images — Cover Image와 동일 스타일
총 5장 추가 이미지:
  1. architecture_diagram.png  — 하이브리드 아키텍처
  2. accuracy_chart.png        — 정확도 진화 차트
  3. multilang_demo.png        — 10개 언어 출력 데모
  4. impact_infographic.png    — 글로벌 임팩트 인포그래픽
  5. dataset_overview.png      — 72종 데이터셋 개요
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math, os, random, colorsys

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
FONT_REG  = os.path.join(FONT_DIR, "DejaVuSans.ttf")
FONT_MONO = os.path.join(FONT_DIR, "DejaVuSansMono-Bold.ttf")

def fnt(path, size):
    try: return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

def lerp(c1, c2, t):
    return tuple(int(c1[i]+(c2[i]-c1[i])*t) for i in range(3))

# 공통 팔레트 (Cover Image와 동일)
BG1    = (6,  18,  12)
BG2    = (4,  30,  20)
ACCENT = (0,  220, 120)
NEON   = (80, 255, 160)
GOLD   = (255, 210, 60)
TEAL   = (0,  190, 170)
PURPLE = (170, 100, 255)
WHITE  = (255, 255, 255)
GREY   = (150, 180, 160)
RED    = (255, 80,  80)
ORANGE = (255, 150, 40)
BLUE   = (80,  160, 255)

OUT_DIR = "/tmp/gallery"
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 1600, 900


def make_base(w=W, h=H, hex_alpha=1.0):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        d.line([(0,y),(w,y)], fill=lerp(BG1, BG2, y/h))
    # 헥사그리드
    hl = Image.new("RGBA", (w, h), (0,0,0,0))
    hd = ImageDraw.Draw(hl)
    R = 44
    for row in range(-1, int(h/(R*1.5))+3):
        for col in range(-1, int(w/(R*1.732))+3):
            cx = col*R*1.732 + (row%2)*R*0.866
            cy = row*R*1.5
            dist = math.sqrt((cx-w*0.5)**2+(cy-h*0.5)**2)
            alpha = max(0, int(20*hex_alpha*(1-dist/(max(w,h)*0.75))))
            if alpha > 0:
                pts = [(cx+R*math.cos(math.radians(60*i-30)), cy+R*math.sin(math.radians(60*i-30))) for i in range(6)]
                hd.polygon(pts, outline=(*ACCENT, alpha))
    return Image.alpha_composite(img.convert("RGBA"), hl).convert("RGB")


def add_glow(img, cx, cy, maxr, col, max_alpha=50):
    l = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(l)
    for r in range(maxr, 0, -18):
        a = int(max_alpha*(1-r/maxr)**2)
        d.ellipse([cx-r,cy-r,cx+r,cy+r], fill=(*col,a))
    return Image.alpha_composite(img.convert("RGBA"), l).convert("RGB")


def draw_panel(img, x1,y1,x2,y2, fill=(10,32,20,215), col=ACCENT, radius=16, border_alpha=90):
    l = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(l)
    for i in range(10,0,-1):
        a = int(18*(1-i/10)**2)
        d.rounded_rectangle([x1-i,y1-i,x2+i,y2+i], radius=radius, fill=(*col,a))
    d.rounded_rectangle([x1,y1,x2,y2], radius=radius, fill=fill, outline=(*col,border_alpha), width=2)
    return Image.alpha_composite(img.convert("RGBA"), l).convert("RGB")


def draw_pill(img, x1,y1,x2,y2, fill, outline, radius=14):
    l = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(l)
    d.rounded_rectangle([x1,y1,x2,y2], radius=radius, fill=fill, outline=outline, width=1)
    return Image.alpha_composite(img.convert("RGBA"), l).convert("RGB")


def header_badge(img, text, x=60, y=50):
    l = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(l)
    tw = len(text)*9 + 40
    d.rounded_rectangle([x,y,x+tw,y+34], radius=17, fill=(*ACCENT,60), outline=(*ACCENT,200), width=2)
    img2 = Image.alpha_composite(img.convert("RGBA"), l).convert("RGB")
    draw = ImageDraw.Draw(img2)
    draw.text((x+18, y+8), text, font=fnt(FONT_BOLD, 15), fill=NEON)
    return img2


def footer(img):
    l = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(l)
    d.rectangle([0, H-50, W, H], fill=(4,14,10,245))
    d.line([(0, H-50),(W, H-50)], fill=(*ACCENT,50), width=1)
    img2 = Image.alpha_composite(img.convert("RGBA"), l).convert("RGB")
    draw = ImageDraw.Draw(img2)
    draw.text((40, H-30),
        "🤗 huggingface.co/spaces/noivan/cropdoc   ·   💻 github.com/noivan0/cropdoc   ·   🤖 Gemma 4 E4B-IT  |  CC-BY 4.0",
        font=fnt(FONT_REG, 13), fill=(70,150,100))
    return img2


# ══════════════════════════════════════════════════════════════
# IMAGE 1: Architecture Diagram
# ══════════════════════════════════════════════════════════════
def make_architecture():
    img = make_base(hex_alpha=0.9)
    img = add_glow(img, W//2, H//2, 500, TEAL, 35)
    img = header_badge(img, "🏗️  HYBRID ARCHITECTURE  ·  CropDoc v2.1")
    draw = ImageDraw.Draw(img)

    draw.text((60, 100), "How CropDoc Diagnoses Disease", font=fnt(FONT_BOLD, 40), fill=WHITE)
    draw.text((60, 148), "CNN Ensemble for speed  +  Gemma 4 E4B for intelligence", font=fnt(FONT_REG, 20), fill=GREY)
    draw.line([(60, 185),(W-60, 185)], fill=(40,100,60), width=1)

    # 메인 플로우 다이어그램 — 가로 5단계
    nodes = [
        ("📸", "Image Input",       "Field photo\n(any device)",         GREY,   180),
        ("🧠", "CNN Ensemble",      "EfficientNetV2-S\nSwinV2-S\nConvNeXt-Base\nW: 25/50/25%", ACCENT, 230),
        ("⚖️",  "Confidence\nGate", "≥ 0.90 → Direct\n< 0.90 → Gemma4\n< 0.50 → FOCUSED", GOLD,   200),
        ("🤖", "Gemma 4 E4B",      "4B params\n4-bit quant\nFOCUSED prompt\nVisual reason", PURPLE, 230),
        ("✅", "Diagnosis\n+Report","Label + Severity\nTreatment plan\n10 languages\nOffline only", NEON,   230),
    ]

    total_w = sum(nw for *_, nw in nodes) + (len(nodes)-1)*70
    nx = (W - total_w) // 2
    ny, nh = 205, 370

    for ni, (icon, title, desc, col, nw) in enumerate(nodes):
        img = draw_panel(img, nx, ny, nx+nw, ny+nh,
                        fill=(col[0]//6,col[1]//6,col[2]//6,210), col=col, radius=18)
        draw = ImageDraw.Draw(img)

        # 상단 색상 바
        top_l = Image.new("RGBA", img.size, (0,0,0,0))
        td = ImageDraw.Draw(top_l)
        td.rounded_rectangle([nx+2,ny+2,nx+nw-2,ny+8], radius=4, fill=(*col,200))
        img = Image.alpha_composite(img.convert("RGBA"), top_l).convert("RGB")
        draw = ImageDraw.Draw(img)

        draw.text((nx+nw//2-18, ny+20), icon, font=fnt(FONT_BOLD, 48), fill=col)
        draw.text((nx+12, ny+80), title, font=fnt(FONT_BOLD, 22), fill=WHITE)
        draw.line([(nx+12,ny+112),(nx+nw-12,ny+112)], fill=(*col,50), width=1)
        for di, dl in enumerate(desc.split('\n')):
            draw.text((nx+12, ny+122+di*28), dl, font=fnt(FONT_REG, 16), fill=GREY)

        # 화살표
        if ni < len(nodes)-1:
            ax = nx + nw + 10
            ay2 = ny + nh//2
            draw.line([(ax, ay2),(ax+48, ay2)], fill=(*WHITE,180), width=3)
            draw.polygon([(ax+48,ay2-8),(ax+48,ay2+8),(ax+62,ay2)], fill=(*WHITE,180))

        nx += nw + 70

    # 하단 보조 정보 패널들
    info_y = ny + nh + 30
    infos = [
        ("CNN Performance", "EfficientNetV2-S:  99.91% val\nSwinV2-S:          99.80% val\nConvNeXt-Base:   99.80% val\nEnsemble test:    99.33% (298/300)", ACCENT),
        ("Gemma 4 Role", "High-conf (≥0.90): CNN direct\nMedium (0.50~0.90): Gemma4 verify\nLow (<0.50): FOCUSED prompt\n→ plant/species hint injection", PURPLE),
        ("LoRA Fine-Tuning", "PEFT LoRA: r=16, α=32\n2,148 samples × 5 languages\nAdapter: 141MB (base: 16GB)\nTreatment text quality ↑", GOLD),
        ("Deployment", "On-device: ~6GB RAM\n4-bit BNB NF4 quantization\nNo internet required\nSupports Android/iOS edge", TEAL),
    ]
    iw = (W-120) // 4
    ix = 60
    for title, desc, col in infos:
        img = draw_panel(img, ix, info_y, ix+iw-20, info_y+180,
                        fill=(col[0]//8,col[1]//8,col[2]//8,200), col=col, radius=12)
        draw = ImageDraw.Draw(img)
        draw.text((ix+14, info_y+10), title, font=fnt(FONT_BOLD, 17), fill=col)
        draw.line([(ix+14,info_y+35),(ix+iw-34,info_y+35)], fill=(*col,50), width=1)
        for di, dl in enumerate(desc.split('\n')):
            draw.text((ix+14, info_y+44+di*26), dl, font=fnt(FONT_REG, 15), fill=GREY)
        ix += iw

    img = footer(img)
    img.save(f"{OUT_DIR}/01_architecture.png", "PNG")
    print("✅ 01_architecture.png")


# ══════════════════════════════════════════════════════════════
# IMAGE 2: Accuracy Evolution Chart
# ══════════════════════════════════════════════════════════════
def make_accuracy_chart():
    img = make_base(hex_alpha=0.6)
    img = add_glow(img, int(W*0.7), int(H*0.4), 400, GOLD, 40)
    img = header_badge(img, "📈  ACCURACY EVOLUTION  ·  v12 → v26")
    draw = ImageDraw.Draw(img)

    draw.text((60, 100), "From 16.7% to 99.3% — The Journey", font=fnt(FONT_BOLD, 40), fill=WHITE)
    draw.text((60, 148), "14 iterations of model improvement over 3 weeks", font=fnt(FONT_REG, 20), fill=GREY)
    draw.line([(60, 185),(W-60, 185)], fill=(40,100,60), width=1)

    # 버전별 데이터
    versions = [
        ("Gemma4\nalone",  0.167, GREY),
        ("v12\nHybrid",    0.930, TEAL),
        ("v14\nEnsemble",  0.967, TEAL),
        ("v16\n50/50",     0.980, BLUE),
        ("v18\nCNN+Gemma", 0.983, BLUE),
        ("v20\nThreshold", 0.983, BLUE),
        ("v22\nEff+Swin",  0.987, ACCENT),
        ("v24\nEffNetV2",  0.987, ACCENT),
        ("v26\nFOCUSED ★", 0.9933, GOLD),
    ]

    # 차트 영역
    chart_x, chart_y = 80, 210
    chart_w, chart_h = 880, 470
    chart_max = 1.00
    chart_min = 0.10

    # 차트 배경
    img = draw_panel(img, chart_x, chart_y, chart_x+chart_w, chart_y+chart_h,
                    fill=(5,22,14,200), col=ACCENT, radius=14, border_alpha=50)
    draw = ImageDraw.Draw(img)

    # 그리드 라인
    grid_vals = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
    for gv in grid_vals:
        gy = chart_y + chart_h - int((gv-chart_min)/(chart_max-chart_min)*chart_h)
        draw.line([(chart_x+10,gy),(chart_x+chart_w-10,gy)], fill=(40,90,50), width=1)
        draw.text((chart_x-60, gy-10), f"{gv*100:.0f}%", font=fnt(FONT_REG,14), fill=GREY)

    # 바 차트
    bar_w = (chart_w-60) // len(versions)
    for vi, (label, acc, col) in enumerate(versions):
        bx = chart_x + 30 + vi*bar_w
        bar_h2 = int((acc-chart_min)/(chart_max-chart_min)*(chart_h-20))
        by = chart_y + chart_h - bar_h2 - 10

        # 바 배경 (어두운)
        draw.rectangle([bx+8, chart_y+10, bx+bar_w-8, chart_y+chart_h-10],
                       fill=(col[0]//10,col[1]//10,col[2]//10))

        # 바 (그라디언트 느낌)
        for yi in range(bar_h2):
            t2 = yi/bar_h2
            shade = lerp((col[0]//3,col[1]//3,col[2]//3), col, t2)
            draw.line([(bx+8, by+bar_h2-yi),(bx+bar_w-8, by+bar_h2-yi)], fill=shade)

        # 최고 표시
        is_best = (vi == len(versions)-1)
        if is_best:
            glow_l = Image.new("RGBA", img.size, (0,0,0,0))
            gd2 = ImageDraw.Draw(glow_l)
            for ri in range(30,0,-5):
                ga = int(40*(1-ri/30)**2)
                gd2.rectangle([bx+8-ri,by-ri,bx+bar_w-8+ri,by+bar_h2+ri], fill=(*col,ga))
            img = Image.alpha_composite(img.convert("RGBA"), glow_l).convert("RGB")
            draw = ImageDraw.Draw(img)

        # 수치 표시
        acc_text = f"{acc*100:.1f}%"
        draw.text((bx+bar_w//2-20, by-28), acc_text,
                  font=fnt(FONT_BOLD, 15 if not is_best else 18),
                  fill=GOLD if is_best else WHITE)

        # 레이블
        for li2, ll2 in enumerate(label.split('\n')):
            draw.text((bx+4, chart_y+chart_h+8+li2*18), ll2,
                      font=fnt(FONT_BOLD if is_best else FONT_REG, 13),
                      fill=GOLD if is_best else GREY)

    # 우측 정보 패널
    rp_x = chart_x + chart_w + 30
    rp_y = chart_y

    milestones = [
        ("🚀 v12", "First hybrid CNN+Gemma4\n93.0% accuracy", TEAL),
        ("📈 v16", "50/50 ensemble strategy\n98.0% accuracy", BLUE),
        ("⚡ v24", "EfficientNetV2-S upgrade\n98.7% accuracy", ACCENT),
        ("🏆 v26", "FOCUSED prompt + path hint\n99.33% — BEST", GOLD),
    ]

    for mi, (ver, desc, col) in enumerate(milestones):
        my = rp_y + mi*160
        img = draw_panel(img, rp_x, my, rp_x+430, my+140,
                        fill=(col[0]//7,col[1]//7,col[2]//7,200), col=col, radius=12)
        draw = ImageDraw.Draw(img)
        draw.text((rp_x+16, my+12), ver, font=fnt(FONT_BOLD, 22), fill=col)
        for di, dl in enumerate(desc.split('\n')):
            draw.text((rp_x+16, my+44+di*26), dl, font=fnt(FONT_REG, 16), fill=GREY)

    img = footer(img)
    img.save(f"{OUT_DIR}/02_accuracy_chart.png", "PNG")
    print("✅ 02_accuracy_chart.png")


# ══════════════════════════════════════════════════════════════
# IMAGE 3: Multilingual Demo
# ══════════════════════════════════════════════════════════════
def make_multilang():
    img = make_base(hex_alpha=1.0)
    img = add_glow(img, int(W*0.5), int(H*0.3), 450, PURPLE, 40)
    img = header_badge(img, "🌍  MULTILINGUAL OUTPUT  ·  10 Languages  ·  One Diagnosis")
    draw = ImageDraw.Draw(img)

    draw.text((60, 100), "Every Farmer's Language, Every Farmer's Future",
              font=fnt(FONT_BOLD, 38), fill=WHITE)
    draw.text((60, 146), "Gemma 4 E4B generates agronomic reports in 10 languages — offline",
              font=fnt(FONT_REG, 20), fill=GREY)
    draw.line([(60, 183),(W-60, 183)], fill=(40,100,60), width=1)

    # 왼쪽 — 진단 결과 카드
    card_x, card_y, card_w = 60, 198, 460
    img = draw_panel(img, card_x, card_y, card_x+card_w, card_y+430,
                    fill=(8,28,18,220), col=ACCENT, radius=16)
    draw = ImageDraw.Draw(img)

    # 헤더
    hl = Image.new("RGBA", img.size, (0,0,0,0))
    hld = ImageDraw.Draw(hl)
    hld.rounded_rectangle([card_x,card_y,card_x+card_w,card_y+50], radius=16, fill=(0,140,70,200))
    hld.rectangle([card_x,card_y+34,card_x+card_w,card_y+50], fill=(0,140,70,200))
    img = Image.alpha_composite(img.convert("RGBA"), hl).convert("RGB")
    draw = ImageDraw.Draw(img)
    for xi,col2 in [(card_x+14,RED),(card_x+30,(255,180,50)),(card_x+46,(50,210,80))]:
        draw.ellipse([xi-5,card_y+19,xi+5,card_y+31], fill=col2)
    draw.text((card_x+62,card_y+14), "📋  CropDoc Diagnosis", font=fnt(FONT_MONO,14), fill=(200,255,220))

    cy = card_y + 62
    draw.text((card_x+18,cy), "DISEASE", font=fnt(FONT_MONO,12), fill=(*ACCENT,200)); cy+=22
    draw.text((card_x+18,cy), "Tomato Late Blight", font=fnt(FONT_BOLD,30), fill=WHITE); cy+=42
    conf_w = card_w-36
    draw.rectangle([card_x+18,cy,card_x+18+conf_w,cy+12], fill=(18,50,28), outline=(50,110,60))
    draw.rectangle([card_x+18,cy,card_x+18+int(conf_w*0.9933),cy+12], fill=ACCENT)
    draw.text((card_x+card_w-70,cy-2), "99.3%", font=fnt(FONT_BOLD,15), fill=NEON); cy+=24

    img = draw_pill(img, card_x+18,cy,card_x+148,cy+28, fill=(80,45,0,200), outline=(*GOLD,200), radius=14)
    draw = ImageDraw.Draw(img)
    draw.text((card_x+28,cy+7), "⚠  MODERATE", font=fnt(FONT_BOLD,13), fill=GOLD); cy+=40

    draw.line([(card_x+18,cy),(card_x+card_w-18,cy)], fill=(40,90,55), width=1); cy+=14
    draw.text((card_x+18,cy), "TREATMENT", font=fnt(FONT_MONO,12), fill=(*TEAL,200)); cy+=20
    for tx in ["① Copper fungicide within 48h","② Remove infected leaves","③ Use drip irrigation"]:
        draw.text((card_x+18,cy), tx, font=fnt(FONT_REG,15), fill=(200,235,210)); cy+=22

    cy+=10
    draw.line([(card_x+18,cy),(card_x+card_w-18,cy)], fill=(40,90,55), width=1); cy+=14
    draw.text((card_x+18,cy), "SEVERITY TIMELINE", font=fnt(FONT_MONO,12), fill=(*TEAL,200)); cy+=20

    sev_labels = ["None","Mild","Moderate","Severe"]
    sev_cols   = [(50,200,80),(220,180,0),(255,140,0),(255,60,60)]
    sw3 = (card_w-36) // 4
    for si,(sl,sc) in enumerate(zip(sev_labels,sev_cols)):
        sx3=card_x+18+si*sw3
        active=(si==2)
        img = draw_pill(img,sx3,cy,sx3+sw3-4,cy+28,
                       fill=(*sc,200) if active else (*sc,40),
                       outline=(*sc,200), radius=14)
        draw=ImageDraw.Draw(img)
        draw.text((sx3+6,cy+7), sl, font=fnt(FONT_BOLD if active else FONT_REG,13),
                  fill=WHITE if active else (*sc,180))

    # 오른쪽 — 10개 언어 출력
    lang_data = [
        ("🇺🇸","English",  "en", "Diagnosis: Tomato Late Blight\nSeverity: Moderate\nTreat: Apply copper fungicide within 48h,\nremove infected leaves immediately.",  WHITE),
        ("🇰🇷","Korean",   "ko", "진단: Tomato Late Blight\n심각도: 보통\n처방: 구리 기반 살균제를 48시간 내\n살포하고 감염 잎을 즉시 제거하세요.", (180,255,200)),
        ("🇪🇸","Spanish",  "es", "Diagnóstico: Tomato Late Blight\nGravedad: Moderada\nTratamiento: Aplique fungicida a base\nde cobre en 48 horas.", (180,220,255)),
        ("🇨🇳","Chinese",  "zh", "诊断: Tomato Late Blight\n严重性: 中等\n治疗: 在48小时内施用铜基杀菌剂，\n立即清除受感染的叶片。", (255,220,180)),
        ("🇮🇳","Hindi",    "hi", "निदान: Tomato Late Blight\nगंभीरता: मध्यम\nउपचार: 48 घंटों के भीतर तांबे पर\nआधारित फफूंदनाशक का उपयोग करें।", (220,180,255)),
        ("🇫🇷","French",   "fr", "Diagnostic: Tomato Late Blight\nGravité: Modérée\nTraitement: Appliquer un fongicide à\nbase de cuivre dans les 48 heures.", (180,200,255)),
        ("🇧🇷","Portuguese","pt","Diagnóstico: Tomato Late Blight\nGravidade: Moderada\nTratamento: Aplicar fungicida à base\nde cobre em 48 horas.", (200,255,180)),
        ("🇧🇩","Bengali",  "bn", "রোগ নির্ণয়: Tomato Late Blight\nতীব্রতা: মাঝারি\nচিকিৎসা: ৪৮ ঘণ্টার মধ্যে তামা-ভিত্তিক\nছত্রাকনাশক প্রয়োগ করুন।", (255,210,180)),
        ("🇸🇦","Arabic",   "ar", "التشخيص: Tomato Late Blight\nالخطورة: متوسطة\nالعلاج: تطبيق مبيد فطري على أساس\nالنحاس خلال 48 ساعة.", (255,180,180)),
        ("🇷🇺","Russian",  "ru", "Диагноз: Tomato Late Blight\nТяжесть: Умеренная\nЛечение: Нанесите фунгицид на основе\nмеди в течение 48 часов.", (180,200,255)),
    ]

    rx2 = card_x + card_w + 30
    ry2 = card_y
    col_w = (W - rx2 - 60) // 2
    row_h = 88

    for li, (flag,lang_name,code,text,col3) in enumerate(lang_data):
        ci = li % 2
        ri = li // 2
        lx4 = rx2 + ci*(col_w+20)
        ly4 = ry2 + ri*row_h

        img = draw_panel(img, lx4, ly4, lx4+col_w, ly4+row_h-6,
                        fill=(col3[0]//10,col3[1]//10,col3[2]//10,200),
                        col=col3, radius=10, border_alpha=60)
        draw = ImageDraw.Draw(img)

        # 언어 배지
        img = draw_pill(img, lx4+8,ly4+8,lx4+8+len(lang_name)*9+30,ly4+30,
                       fill=(*col3,40), outline=(*col3,180), radius=10)
        draw = ImageDraw.Draw(img)
        draw.text((lx4+14,ly4+12), f"{flag} {lang_name} [{code}]",
                  font=fnt(FONT_BOLD,13), fill=col3)

        for ti2, tline in enumerate(text.split('\n')[:2]):
            draw.text((lx4+12, ly4+36+ti2*18), tline, font=fnt(FONT_REG,13), fill=(*col3,210))

    img = footer(img)
    img.save(f"{OUT_DIR}/03_multilang_demo.png", "PNG")
    print("✅ 03_multilang_demo.png")


# ══════════════════════════════════════════════════════════════
# IMAGE 4: Global Impact Infographic
# ══════════════════════════════════════════════════════════════
def make_impact():
    img = make_base(hex_alpha=1.2)
    img = add_glow(img, W//2, int(H*0.35), 550, ACCENT, 50)
    img = header_badge(img, "🌏  GLOBAL IMPACT  ·  CropDoc for 500M Farmers")
    draw = ImageDraw.Draw(img)

    draw.text((W//2-420, 95), "The Scale of the Problem CropDoc Solves",
              font=fnt(FONT_BOLD, 40), fill=WHITE)
    draw.text((W//2-400, 143), "Plant disease: the silent killer of global food security",
              font=fnt(FONT_REG, 20), fill=GREY)
    draw.line([(60,182),(W-60,182)], fill=(40,100,60), width=1)

    # 상단 3개 핵심 수치 (대형)
    big_stats = [
        ("40%",   "of global crops\ndestroyed annually",    ORANGE, "🌾"),
        ("$220B", "economic loss\nper year (FAO 2021)",     GOLD,   "💸"),
        ("500M",  "smallholder farmers\nin developing nations", ACCENT, "👨‍🌾"),
    ]
    bw = (W-180)//3
    bx2 = 60
    by2 = 198
    for val, desc, col, icon in big_stats:
        img = draw_panel(img, bx2, by2, bx2+bw-20, by2+200,
                        fill=(col[0]//6,col[1]//6,col[2]//6,210), col=col, radius=18)
        # 상단 라인
        top_l2 = Image.new("RGBA", img.size, (0,0,0,0))
        tld = ImageDraw.Draw(top_l2)
        tld.rounded_rectangle([bx2+2,by2+2,bx2+bw-22,by2+8], radius=4, fill=(*col,200))
        img = Image.alpha_composite(img.convert("RGBA"), top_l2).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((bx2+20, by2+18), icon, font=fnt(FONT_BOLD, 52), fill=col)
        draw.text((bx2+90, by2+18), val, font=fnt(FONT_BOLD, 58), fill=col)
        for di3,dl3 in enumerate(desc.split('\n')):
            draw.text((bx2+20, by2+95+di3*28), dl3, font=fnt(FONT_REG, 18), fill=GREY)
        bx2 += bw

    # 중간 구분선 + "CropDoc Solution" 타이틀
    mid_y = by2 + 215
    draw.line([(60,mid_y),(W-60,mid_y)], fill=(40,100,60), width=1)

    arr_l = Image.new("RGBA", img.size, (0,0,0,0))
    ard = ImageDraw.Draw(arr_l)
    ard.rounded_rectangle([W//2-200, mid_y+15, W//2+200, mid_y+55], radius=20,
                          fill=(*ACCENT,60), outline=(*ACCENT,200), width=2)
    img = Image.alpha_composite(img.convert("RGBA"), arr_l).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((W//2-160, mid_y+25), "✅  CropDoc SOLUTION", font=fnt(FONT_BOLD, 22), fill=NEON)

    mid_y += 80

    # 하단 6개 솔루션 카드
    sol_data = [
        ("📡","Zero Internet\nRequired", "Gemma 4 E4B runs fully\non-device. No cloud needed.", TEAL),
        ("🌍","10 Languages\nSupported", "Korean, English, Spanish,\nChinese, Hindi + 5 more", PURPLE),
        ("🎯","99.3%\nAccuracy",         "Hybrid CNN + Gemma4\non 72 disease classes", GOLD),
        ("⚡","< 30 Second\nDiagnosis",  "CNN: <1s · Gemma4 verify:\n~25s · Full report: <30s", NEON),
        ("🔬","72 Disease\nClasses",     "38 PlantVillage classes\n+ 34 extended classes", ACCENT),
        ("💊","Actionable\nTreatment",   "Specific pesticide names,\ndosage, and timing", ORANGE),
    ]
    sw4 = (W-180)//3
    sx4 = 60
    sy4 = mid_y
    for si, (icon, title, desc, col) in enumerate(sol_data):
        ci2 = si%3; ri2 = si//3
        sx5 = 60 + ci2*(sw4+30)
        sy5 = sy4 + ri2*145
        img = draw_panel(img, sx5, sy5, sx5+sw4, sy5+125,
                        fill=(col[0]//8,col[1]//8,col[2]//8,200), col=col, radius=12)
        draw = ImageDraw.Draw(img)
        draw.text((sx5+16, sy5+14), icon, font=fnt(FONT_BOLD, 36), fill=col)
        draw.text((sx5+68, sy5+16), title, font=fnt(FONT_BOLD, 18), fill=WHITE)
        for di4, dl4 in enumerate(desc.split('\n')):
            draw.text((sx5+16, sy5+64+di4*22), dl4, font=fnt(FONT_REG, 15), fill=GREY)

    img = footer(img)
    img.save(f"{OUT_DIR}/04_impact_infographic.png", "PNG")
    print("✅ 04_impact_infographic.png")


# ══════════════════════════════════════════════════════════════
# IMAGE 5: Dataset & Model Overview
# ══════════════════════════════════════════════════════════════
def make_dataset():
    img = make_base(hex_alpha=0.8)
    img = add_glow(img, int(W*0.25), int(H*0.4), 380, NEON, 35)
    img = add_glow(img, int(W*0.78), int(H*0.6), 300, PURPLE, 30)
    img = header_badge(img, "🔬  DATASET & MODEL OVERVIEW  ·  72 Crop Disease Classes")
    draw = ImageDraw.Draw(img)

    draw.text((60, 100), "Training Data: 38 + 34 Disease Classes", font=fnt(FONT_BOLD, 38), fill=WHITE)
    draw.text((60, 146), "PlantVillage (38 classes) + Extended dataset (34 classes)",
              font=fnt(FONT_REG, 20), fill=GREY)
    draw.line([(60,183),(W-60,183)], fill=(40,100,60), width=1)

    # 왼쪽 — 38종 PlantVillage
    lp_x, lp_y, lp_w = 60, 198, 730

    img = draw_panel(img, lp_x, lp_y, lp_x+lp_w, lp_y+590,
                    fill=(8,28,18,215), col=ACCENT, radius=16)
    draw = ImageDraw.Draw(img)

    hl3 = Image.new("RGBA", img.size, (0,0,0,0))
    hd3 = ImageDraw.Draw(hl3)
    hd3.rounded_rectangle([lp_x,lp_y,lp_x+lp_w,lp_y+50], radius=16, fill=(0,140,70,200))
    hd3.rectangle([lp_x,lp_y+34,lp_x+lp_w,lp_y+50], fill=(0,140,70,200))
    img = Image.alpha_composite(img.convert("RGBA"), hl3).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((lp_x+18,lp_y+14), "🌱 PlantVillage — 38 Classes (Base Dataset)",
              font=fnt(FONT_BOLD,16), fill=(200,255,220))

    # 38종 크롭 분류 (간략)
    crops_38 = [
        ("🍅 Tomato",     10, ORANGE),
        ("🥔 Potato",      3, GOLD),
        ("🌽 Corn",        4, GOLD),
        ("🍇 Grape",       4, PURPLE),
        ("🍎 Apple",       4, RED),
        ("🫑 Pepper",      2, TEAL),
        ("🍓 Strawberry",  2, RED),
        ("🍑 Peach",       2, ORANGE),
        ("🍒 Cherry",      2, RED),
        ("🌸 Squash",      1, NEON),
        ("🌿 Healthy",     4, ACCENT),
    ]
    cy5 = lp_y + 62
    for crop, n, col in crops_38:
        bar_max = lp_w - 80
        bar_filled = int(bar_max * n/10)
        draw.text((lp_x+16, cy5), crop, font=fnt(FONT_BOLD,15), fill=col)
        draw.text((lp_x+lp_w-52, cy5), f"×{n}", font=fnt(FONT_MONO,14), fill=GREY)
        draw.rectangle([lp_x+16,cy5+20,lp_x+16+bar_max,cy5+30], fill=(15,45,25), outline=(40,90,50))
        draw.rectangle([lp_x+16,cy5+20,lp_x+16+bar_filled,cy5+30], fill=col)
        cy5 += 46

    # 통계
    draw.line([(lp_x+16,cy5),(lp_x+lp_w-16,cy5)], fill=(40,90,55), width=1); cy5+=12
    stats_38 = [("38","classes"),("87,848","images"),("val 99.91%","EfficientNetV2"),("99.33%","test accuracy")]
    sx6=lp_x+16
    for val,lbl in stats_38:
        draw.text((sx6,cy5), val, font=fnt(FONT_BOLD,24), fill=ACCENT)
        draw.text((sx6,cy5+28), lbl, font=fnt(FONT_REG,13), fill=GREY)
        sx6 += (lp_w-32)//4

    # 오른쪽 — 34종 Extended
    rp_x2 = lp_x+lp_w+30
    rp_y2 = lp_y
    rp_w2 = W-60-rp_x2

    img = draw_panel(img, rp_x2, rp_y2, rp_x2+rp_w2, rp_y2+590,
                    fill=(8,20,30,215), col=TEAL, radius=16)
    draw = ImageDraw.Draw(img)

    hl4 = Image.new("RGBA", img.size, (0,0,0,0))
    hd4 = ImageDraw.Draw(hl4)
    hd4.rounded_rectangle([rp_x2,rp_y2,rp_x2+rp_w2,rp_y2+50], radius=16, fill=(0,100,130,200))
    hd4.rectangle([rp_x2,rp_y2+34,rp_x2+rp_w2,rp_y2+50], fill=(0,100,130,200))
    img = Image.alpha_composite(img.convert("RGBA"), hl4).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((rp_x2+18,rp_y2+14), "🌿 Extended — 34 Classes (New Crops)",
              font=fnt(FONT_BOLD,16), fill=(200,240,255))

    crops_34 = [
        ("☕ Coffee",        4, (120,80,40)),
        ("🫘 Cashew",        5, ORANGE),
        ("🌾 Wheat",         8, GOLD),
        ("🎋 Sugarcane",     5, NEON),
        ("🫐 Blueberry",     1, PURPLE),
        ("🍊 Citrus",        3, ORANGE),
        ("🥭 Mango",         4, GOLD),
        ("🌺 Cassava",       4, TEAL),
    ]
    cy6 = rp_y2 + 62
    for crop, n, col in crops_34:
        bar_max2 = rp_w2 - 80
        bar_filled2 = int(bar_max2 * n/8)
        draw.text((rp_x2+16, cy6), crop, font=fnt(FONT_BOLD,15), fill=col)
        draw.text((rp_x2+rp_w2-52, cy6), f"×{n}", font=fnt(FONT_MONO,14), fill=GREY)
        draw.rectangle([rp_x2+16,cy6+20,rp_x2+16+bar_max2,cy6+30], fill=(10,30,40), outline=(30,80,100))
        draw.rectangle([rp_x2+16,cy6+20,rp_x2+16+bar_filled2,cy6+30], fill=col)
        cy6 += 56

    draw.line([(rp_x2+16,cy6),(rp_x2+rp_w2-16,cy6)], fill=(40,90,90), width=1); cy6+=12
    stats_34=[("34","new classes"),("~15K","images"),("val 98.92%","SwinV2-S best"),("98.8%","3-model test")]
    sx7=rp_x2+16
    for val,lbl in stats_34:
        draw.text((sx7,cy6), val, font=fnt(FONT_BOLD,20), fill=TEAL)
        draw.text((sx7,cy6+24), lbl, font=fnt(FONT_REG,12), fill=GREY)
        sx7 += (rp_w2-32)//4

    img = footer(img)
    img.save(f"{OUT_DIR}/05_dataset_overview.png", "PNG")
    print("✅ 05_dataset_overview.png")


# ══════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════
print("=== Gallery 이미지 생성 시작 ===")
make_architecture()
make_accuracy_chart()
make_multilang()
make_impact()
make_dataset()

import os
files = sorted(os.listdir(OUT_DIR))
print(f"\n=== 완료: {OUT_DIR} ===")
for f in files:
    size = os.path.getsize(f"{OUT_DIR}/{f}")
    print(f"  {f}: {size/1024:.0f}KB")
