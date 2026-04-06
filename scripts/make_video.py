"""
CropDoc YouTube Demo Video — 최적화 버전
해상도: 1280×720 (HD), 12fps, 총 ~150초 (2분 30초)
렌더링 시간: ~3분 예상
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import math, os, random
import imageio
import imageio_ffmpeg

W, H = 1280, 720

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
FONT_REG  = os.path.join(FONT_DIR, "DejaVuSans.ttf")
FONT_MONO = os.path.join(FONT_DIR, "DejaVuSansMono-Bold.ttf")

def fnt(path, size):
    try: return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

def lerp(c1, c2, t):
    return tuple(int(c1[i]+(c2[i]-c1[i])*t) for i in range(3))

def ease_out(t): return 1-(1-min(max(t,0),1))**3
def ease_inout(t): t=min(max(t,0),1); return t*t*(3-2*t)

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

# 캐시된 배경
_bg_cache = None
def make_bg():
    global _bg_cache
    if _bg_cache is None:
        img = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(img)
        for y in range(H):
            t = y/H
            d.line([(0,y),(W,y)], fill=lerp(BG1, BG2, t))
        # 육각형 그리드
        hex_l = Image.new("RGBA", (W, H), (0,0,0,0))
        hd = ImageDraw.Draw(hex_l)
        R = 36
        for row in range(-1, int(H/(R*1.5))+3):
            for col in range(-1, int(W/(R*1.732))+3):
                cx = col*R*1.732 + (row%2)*R*0.866
                cy = row*R*1.5
                dist = math.sqrt((cx-W*0.5)**2+(cy-H*0.5)**2)
                alpha = max(0, int(16*(1-dist/(max(W,H)*0.7))))
                if alpha > 0:
                    pts = [(cx+R*math.cos(math.radians(60*i-30)), cy+R*math.sin(math.radians(60*i-30))) for i in range(6)]
                    hd.polygon(pts, outline=(*ACCENT, alpha))
        _bg_cache = Image.alpha_composite(img.convert("RGBA"), hex_l).convert("RGB")
    return _bg_cache.copy()

def add_glow(img, cx, cy, maxr, col, max_alpha=45):
    glow_l = Image.new("RGBA", img.size, (0,0,0,0))
    gd = ImageDraw.Draw(glow_l)
    for r in range(maxr, 0, -20):
        a = int(max_alpha*(1-r/maxr)**2)
        gd.ellipse([cx-r,cy-r,cx+r,cy+r], fill=(*col, a))
    return Image.alpha_composite(img.convert("RGBA"), glow_l).convert("RGB")

def draw_panel(img, x1, y1, x2, y2, fill=(10,32,20,220), col=None, radius=14):
    col = col or ACCENT
    l = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(l)
    for i in range(8,0,-1):
        a = int(15*(1-i/8)**2)
        d.rounded_rectangle([x1-i,y1-i,x2+i,y2+i], radius=radius, fill=(*col,a))
    d.rounded_rectangle([x1,y1,x2,y2], radius=radius, fill=fill, outline=(*col,90), width=2)
    return Image.alpha_composite(img.convert("RGBA"), l).convert("RGB")

def draw_pill(img, x1,y1,x2,y2, fill, outline, r=12):
    l = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(l)
    d.rounded_rectangle([x1,y1,x2,y2], radius=r, fill=fill, outline=outline, width=1)
    return Image.alpha_composite(img.convert("RGBA"), l).convert("RGB")

def alpha_text(img, xy, text, font, color, alpha=255):
    tl = Image.new("RGBA", img.size, (0,0,0,0))
    td = ImageDraw.Draw(tl)
    td.text(xy, text, font=font, fill=(*color[:3], alpha))
    return Image.alpha_composite(img.convert("RGBA"), tl).convert("RGB")

# ════════════════════════════════════════════════════
# SCENE 1: TITLE (0~10s)
# ════════════════════════════════════════════════════
def scene_title(t):
    img = make_bg()
    img = add_glow(img, W//2, H//2, 400, ACCENT, 40)
    draw = ImageDraw.Draw(img)

    prog = ease_out(min(t/2.0, 1.0))
    a = int(255*prog)
    y_off = int((1-prog)*40)

    # 배지
    img = draw_pill(img, W//2-240, 80-y_off, W//2+240, 116-y_off,
                   fill=(*ACCENT,a//3), outline=(*ACCENT,a), r=18)
    draw = ImageDraw.Draw(img)
    draw.text((W//2-190, 90-y_off), "🏆  GEMMA 4 GOOD HACKATHON · 2026",
              font=fnt(FONT_BOLD,15), fill=(*NEON,a))

    # 메인 타이틀
    ty = 145 - y_off
    draw.text((W//2-255+2, ty+2), "CropDoc", font=fnt(FONT_BOLD,110), fill=(0,30,15))
    draw.text((W//2-255, ty), "CropDoc", font=fnt(FONT_BOLD,110), fill=(*NEON,a))

    draw.text((W//2-218, ty+115), "Plant Disease AI", font=fnt(FONT_BOLD,38), fill=(*WHITE,a))
    draw.text((W//2-228, ty+160), "Offline · Multimodal · 10 Languages · 72 Classes",
              font=fnt(FONT_REG,18), fill=(*GREY,a))

    # 수치
    stats = [("500M","Farmers"), ("72","Diseases"), ("99.3%","Accuracy"), ("$220B","Impact")]
    sw = 170
    sx = W//2 - sw*2 + 5
    sy = ty + 205
    cols = [ACCENT, TEAL, GOLD, ORANGE]
    for i,(val,lbl) in enumerate(stats):
        draw.text((sx, sy), val, font=fnt(FONT_BOLD,34), fill=(*cols[i],a))
        draw.text((sx, sy+40), lbl, font=fnt(FONT_REG,16), fill=(*GREY,a))
        sx += sw

    return np.array(img)

# ════════════════════════════════════════════════════
# SCENE 2: PROBLEM (10~28s)
# ════════════════════════════════════════════════════
def scene_problem(t):
    img = make_bg()
    img = add_glow(img, int(W*0.1), int(H*0.8), 250, (200,50,0), 30)
    draw = ImageDraw.Draw(img)

    draw.text((60, 40), "THE PROBLEM", font=fnt(FONT_MONO,18), fill=(*RED,220))
    draw.line([(60,72),(W-60,72)], fill=(80,30,20), width=1)

    problems = [
        ("40%",  "of global crops destroyed annually by disease",    ORANGE, "🌾"),
        ("$220B","economic loss per year  (FAO 2021)",               GOLD,   "💸"),
        ("500M", "smallholder farmers have ZERO expert access",      RED,    "👨‍🌾"),
        ("60%+", "of farms lack reliable internet",                  TEAL,   "📡"),
        ("10+",  "languages spoken — English-only AI fails billions", PURPLE, "🗣"),
    ]

    for pi,(val,desc,col,icon) in enumerate(problems):
        show = ease_out(min((t-pi*0.5)/1.2, 1.0))
        if show <= 0: continue
        y = 90 + pi*112
        x_off = int((1-show)*-100)
        a = int(255*show)
        img = draw_panel(img, 55+x_off, y, W-55+x_off, y+95,
                        fill=(col[0]//8,col[1]//8,col[2]//8,200), col=col, radius=12)
        draw = ImageDraw.Draw(img)
        draw.text((100+x_off, y+12), icon, font=fnt(FONT_BOLD,30), fill=(*col,a))
        draw.text((148+x_off, y+12), val, font=fnt(FONT_BOLD,38), fill=(*col,a))
        draw.text((148+x_off, y+58), desc, font=fnt(FONT_REG,18), fill=(*GREY,a))

    return np.array(img)

# ════════════════════════════════════════════════════
# SCENE 3: SOLUTION (28~48s)
# ════════════════════════════════════════════════════
def scene_solution(t):
    img = make_bg()
    img = add_glow(img, int(W*0.75), int(H*0.3), 280, ACCENT, 40)
    draw = ImageDraw.Draw(img)

    draw.text((55, 35), "THE SOLUTION", font=fnt(FONT_MONO,18), fill=(*ACCENT,220))
    draw.text((55, 65), "CropDoc — AI Doctor for Every Farmer", font=fnt(FONT_BOLD,32), fill=WHITE)
    draw.line([(55,108),(W-55,108)], fill=(40,100,55), width=1)

    features = [
        ("📱", "OFFLINE FIRST",   "No internet required\nGemma 4 E4B on-device\n4-bit quantized (~6GB)",  ACCENT),
        ("🌍", "10 LANGUAGES",    "Korean · English · Spanish\nChinese · Hindi · French\n+4 more",         TEAL),
        ("🎯", "99.3% ACCURACY",  "72 crop disease classes\nHybrid CNN + Gemma 4\n300-image verified",     GOLD),
    ]

    cw = (W-170)//3
    for fi,(icon,title,desc,col) in enumerate(features):
        show = ease_out(min((t-fi*0.6)/1.5, 1.0))
        if show <= 0: continue
        cx = 55 + fi*(cw+25)
        cy = 125
        ch = 550
        a = int(255*show)
        img = draw_panel(img, cx, cy, cx+cw, cy+ch,
                        fill=(col[0]//6,col[1]//6,col[2]//6,200), col=col, radius=16)
        # top accent line
        acc_l = Image.new("RGBA", img.size, (0,0,0,0))
        ad = ImageDraw.Draw(acc_l)
        ad.rounded_rectangle([cx+2,cy+2,cx+cw-2,cy+7], radius=3, fill=(*col,a))
        img = Image.alpha_composite(img.convert("RGBA"), acc_l).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((cx+cw//2-20, cy+25), icon, font=fnt(FONT_BOLD,58), fill=(*col,a))
        draw.text((cx+15, cy+95), title, font=fnt(FONT_BOLD,28), fill=(*col,a))
        draw.line([(cx+15,cy+132),(cx+cw-15,cy+132)], fill=(*col,50), width=1)
        for di,dl in enumerate(desc.split('\n')):
            draw.text((cx+15, cy+145+di*30), dl, font=fnt(FONT_REG,17), fill=(*GREY,a))

    return np.array(img)

# ════════════════════════════════════════════════════
# SCENE 4: DEMO (48~95s)
# ════════════════════════════════════════════════════
def scene_demo(t):
    img = make_bg()
    img = add_glow(img, int(W*0.38), int(H*0.5), 300, ACCENT, 35)
    draw = ImageDraw.Draw(img)

    draw.text((55, 28), "LIVE DEMO — DIAGNOSIS IN ACTION", font=fnt(FONT_MONO,15), fill=(*ACCENT,200))

    # 왼쪽 패널 (이미지 스캔)
    lx, ly, lw, lh = 45, 55, 540, 640
    img = draw_panel(img, lx, ly, lx+lw, ly+lh, fill=(8,28,18,220), col=ACCENT, radius=14)

    # 헤더
    hl = Image.new("RGBA", img.size, (0,0,0,0))
    hd2 = ImageDraw.Draw(hl)
    hd2.rounded_rectangle([lx,ly,lx+lw,ly+46], radius=14, fill=(0,140,70,200))
    hd2.rectangle([lx,ly+30,lx+lw,ly+46], fill=(0,140,70,200))
    img = Image.alpha_composite(img.convert("RGBA"), hl).convert("RGB")
    draw = ImageDraw.Draw(img)
    for xi,col2 in [(lx+12,RED),(lx+28,(255,180,50)),(lx+44,(50,210,80))]:
        draw.ellipse([xi-5,ly+17,xi+5,ly+29], fill=col2)
    draw.text((lx+58, ly+13), "📷  Image Capture — Field Scan", font=fnt(FONT_MONO,13), fill=(200,255,220))

    # 잎사귀 영역
    leaf_y = ly+55
    leaf_h = 430
    scan_bg = Image.new("RGBA", img.size, (0,0,0,0))
    sbd = ImageDraw.Draw(scan_bg)
    sbd.rectangle([lx+10,leaf_y,lx+lw-10,leaf_y+leaf_h], fill=(5,30,18,230))
    img = Image.alpha_composite(img.convert("RGBA"), scan_bg).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 잎사귀
    lc_x, lc_y = lx+lw//2, leaf_y+leaf_h//2+15
    draw.ellipse([lc_x-110,lc_y-135,lc_x+110,lc_y+110], fill=(14,85,38), outline=(0,190,80), width=2)
    draw.line([(lc_x,lc_y-130),(lc_x,lc_y+105)], fill=(0,200,80), width=2)
    for ang in [-32,-18,18,32]:
        a2 = math.radians(ang)
        ex = lc_x+90*math.cos(a2); ey = lc_y+90*math.sin(a2)*0.4-40
        draw.line([(lc_x,lc_y-50),(int(ex),int(ey))], fill=(0,180,70), width=1)
    random.seed(42)
    for _ in range(15):
        bx2=lc_x+random.randint(-85,85); by2=lc_y+random.randint(-110,80); br2=random.randint(5,14)
        draw.ellipse([bx2-br2,by2-br2,bx2+br2,by2+br2], fill=(145,58,12))

    # 스캔 라인
    scan_y = leaf_y + int((t*65) % leaf_h)
    for si in range(0, leaf_h, 3):
        sy2 = leaf_y + si
        base_a = int(8 + 4*math.sin(si*0.25))
        near = abs(si-(t*65%leaf_h))
        if near < 25: base_a = int(35*(1-near/25))
        draw.line([(lx+10,sy2),(lx+lw-10,sy2)], fill=(0,255,100,base_a))
    draw.line([(lx+10,scan_y),(lx+lw-10,scan_y)], fill=(*NEON,70), width=2)
    draw.text((lx+18,leaf_y+8), "🔍 SCANNING...", font=fnt(FONT_MONO,12), fill=NEON)

    # 진행바
    bar_y = leaf_y+leaf_h+8
    prog2 = min(t/6.0, 1.0)*0.87
    draw.rectangle([lx+10,bar_y,lx+lw-10,bar_y+12], fill=(18,50,28), outline=(50,110,60))
    draw.rectangle([lx+10,bar_y,lx+10+int((lw-20)*prog2),bar_y+12], fill=ACCENT)
    draw.text((lx+lw-58,bar_y), f"{int(prog2*100)}%", font=fnt(FONT_BOLD,13), fill=NEON)
    draw.text((lx+12,bar_y+18), "tomato_field_001.jpg  |  1024×768  |  CNN+Gemma4 E4B",
              font=fnt(FONT_REG,12), fill=(70,130,90))

    # 오른쪽 패널 (결과)
    rx, ry2, rw, rh = 605, 55, 635, 640
    img = draw_panel(img, rx, ry2, rx+rw, ry2+rh, fill=(8,28,18,220), col=ACCENT, radius=14)

    rl = Image.new("RGBA", img.size, (0,0,0,0))
    rld = ImageDraw.Draw(rl)
    rld.rounded_rectangle([rx,ry2,rx+rw,ry2+46], radius=14, fill=(0,140,70,200))
    rld.rectangle([rx,ry2+30,rx+rw,ry2+46], fill=(0,140,70,200))
    img = Image.alpha_composite(img.convert("RGBA"), rl).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((rx+18,ry2+13), "📋  Diagnosis Results — CropDoc AI v2.1", font=fnt(FONT_MONO,13), fill=(200,255,220))

    rp = ry2+60
    res_prog = ease_out(min((t-1.5)/3.0, 1.0))
    ra = int(255*res_prog)

    draw.text((rx+18,rp), "DIAGNOSIS", font=fnt(FONT_MONO,12), fill=(*ACCENT,ra))
    rp+=22
    draw.text((rx+18,rp), "Tomato Late Blight", font=fnt(FONT_BOLD,36), fill=(*WHITE,ra))
    rp+=48

    draw.text((rx+18,rp), "Confidence", font=fnt(FONT_REG,14), fill=(*GREY,ra))
    rp+=18
    conf = min(res_prog*1.15, 1.0)*0.9933
    draw.rectangle([rx+18,rp,rx+rw-18,rp+14], fill=(18,50,28), outline=(50,110,60))
    draw.rectangle([rx+18,rp,rx+18+int((rw-36)*conf),rp+14], fill=ACCENT)
    draw.text((rx+rw-72,rp-1), f"{conf*100:.1f}%", font=fnt(FONT_BOLD,16), fill=NEON)
    rp+=28

    img = draw_pill(img,rx+18,rp,rx+160,rp+30, fill=(80,45,0,200), outline=(*GOLD,ra*200//255), r=15)
    draw=ImageDraw.Draw(img)
    draw.text((rx+30,rp+8),"⚠  MODERATE",font=fnt(FONT_BOLD,14),fill=(*GOLD,ra))
    img = draw_pill(img,rx+170,rp,rx+360,rp+30, fill=(50,0,90,200), outline=(*PURPLE,ra*180//255), r=15)
    draw=ImageDraw.Draw(img)
    draw.text((rx+182,rp+8),"🤖 Gemma 4 E4B",font=fnt(FONT_BOLD,14),fill=(*PURPLE,ra))
    rp+=44

    draw.line([(rx+18,rp),(rx+rw-18,rp)], fill=(40,90,55), width=1); rp+=14
    draw.text((rx+18,rp),"TREATMENT PLAN",font=fnt(FONT_MONO,12),fill=(*TEAL,ra)); rp+=22

    treats=["① Apply copper fungicide within 48h",
            "② Remove infected leaves immediately",
            "③ Switch to drip irrigation system",
            "④ Improve field drainage"]
    for ti,tx in enumerate(treats):
        ta=ease_out(min((t-2.5-ti*0.4)/1.0,1.0)); ta=int(255*ta)
        draw.text((rx+18,rp),tx,font=fnt(FONT_REG,16),fill=(*WHITE,ta)); rp+=24
    rp+=6
    draw.line([(rx+18,rp),(rx+rw-18,rp)], fill=(40,90,55), width=1); rp+=14

    draw.text((rx+18,rp),"LANGUAGE",font=fnt(FONT_MONO,12),fill=(*TEAL,ra)); rp+=22
    langs=[("🇺🇸","EN"),("🇰🇷","KO"),("🇪🇸","ES"),("🇧🇷","PT"),("🇨🇳","ZH"),("🇮🇳","HI"),("🇫🇷","FR"),("🇧🇩","BN"),("🇸🇦","AR"),("🇷🇺","RU")]
    lx3=rx+18
    for flag,code in langs:
        active=(code=="KO")
        img=draw_pill(img,lx3,rp,lx3+52,rp+24,
                     fill=(0,100,55,220) if active else (15,50,28,180),
                     outline=(*NEON,ra) if active else (*ACCENT,60), r=12)
        draw=ImageDraw.Draw(img)
        draw.text((lx3+4,rp+5),f"{flag}{code}",font=fnt(FONT_REG,12),
                  fill=(*NEON,ra) if active else (*GREY,int(ra*0.8)))
        lx3+=57
    rp+=36

    draw.line([(rx+18,rp),(rx+rw-18,rp)], fill=(40,90,55), width=1); rp+=14
    draw.text((rx+18,rp),"MULTILINGUAL OUTPUT",font=fnt(FONT_MONO,12),fill=(*TEAL,ra)); rp+=22

    ml_prog=ease_out(min((t-4.0)/2.0,1.0)); mla=int(255*ml_prog)
    mls=[("🇺🇸 EN:","Diagnosis: Tomato Late Blight | Severity: Moderate",WHITE),
         ("🇰🇷 KO:","진단: Tomato Late Blight | 심각도: 보통",( 180,255,200)),
         ("🇪🇸 ES:","Diagnóstico: Tomato Late Blight | Gravedad: Moderada",(180,220,255)),
         ("🇨🇳 ZH:","诊断: Tomato Late Blight | 严重性: 中等",(255,220,180))]
    for flag_t,ml_text,col3 in mls:
        draw.text((rx+20,rp),flag_t,font=fnt(FONT_BOLD,13),fill=(*ACCENT,mla))
        draw.text((rx+95,rp),ml_text,font=fnt(FONT_REG,13),fill=(*col3,mla)); rp+=22
    rp+=8

    img=draw_pill(img,rx+18,rp,rx+200,rp+32,fill=(0,45,85,200),outline=(0,160,230,ra),r=16)
    draw=ImageDraw.Draw(img)
    draw.text((rx+30,rp+9),"📡  OFFLINE-FIRST",font=fnt(FONT_BOLD,14),fill=(80,220,255))
    img=draw_pill(img,rx+210,rp,rx+390,rp+32,fill=(40,0,70,200),outline=(*PURPLE,ra*180//255),r=16)
    draw=ImageDraw.Draw(img)
    draw.text((rx+222,rp+9),"⚡  Edge Ready",font=fnt(FONT_BOLD,14),fill=PURPLE)

    return np.array(img)

# ════════════════════════════════════════════════════
# SCENE 5: TECH (95~125s)
# ════════════════════════════════════════════════════
def scene_tech(t):
    img = make_bg()
    img = add_glow(img, W//2, H//2, 400, TEAL, 28)
    draw = ImageDraw.Draw(img)

    draw.text((55,30),"TECHNOLOGY",font=fnt(FONT_MONO,18),fill=(*TEAL,220))
    draw.text((55,62),"Hybrid CNN Ensemble + Gemma 4 Fine-Tuning",font=fnt(FONT_BOLD,28),fill=WHITE)
    draw.line([(55,100),(W-55,100)],fill=(40,100,60),width=1)

    # 아키텍처 플로우
    arch=[
        ("📸\nImage\nInput",           GREY,   110),
        ("🧠 CNN\nEnsemble\n99.3%",    ACCENT, 180),
        ("conf\n≥0.90?",               GOLD,   120),
        ("🤖 Gemma4\nFOCUSED\nverify", PURPLE, 180),
        ("✅ Report\n10 langs",         NEON,   145),
    ]
    total_w = sum(aw for _,_,aw in arch) + (len(arch)-1)*55
    sx = (W-total_w)//2
    ay,ah = 118,275

    for ai,(label,col,aw) in enumerate(arch):
        show=ease_out(min((t-ai*0.5)/1.5,1.0))
        if show<=0: continue
        a=int(255*show)
        if ai>0:
            ax2=sx-28
            draw.line([(ax2,ay+ah//2),(ax2+20,ay+ah//2)],fill=(*WHITE,a),width=2)
            draw.polygon([(ax2+20,ay+ah//2-6),(ax2+20,ay+ah//2+6),(ax2+34,ay+ah//2)],fill=(*WHITE,a))
        img=draw_panel(img,sx,ay,sx+aw,ay+ah,fill=(col[0]//6,col[1]//6,col[2]//6,200),col=col,radius=12)
        draw=ImageDraw.Draw(img)
        lines_l=label.split('\n')
        ty2=ay+18
        for li2,ll2 in enumerate(lines_l):
            fs2=34 if li2==0 else (18 if li2==1 else 15)
            fc2=col if li2==0 else (WHITE if li2==1 else GREY)
            tw=len(ll2)*(fs2//2); tx2=sx+aw//2-tw//2
            draw.text((tx2,ty2),ll2,font=fnt(FONT_BOLD if li2<=1 else FONT_REG,fs2),fill=(*fc2,a))
            ty2+=fs2+6
        sx+=aw+55

    # 성능 테이블
    ty3=ay+ah+30
    draw.line([(55,ty3),(W-55,ty3)],fill=(40,100,60),width=1); ty3+=14
    draw.text((55,ty3),"PERFORMANCE",font=fnt(FONT_MONO,14),fill=(*TEAL,200)); ty3+=26

    rows=[
        ("Model","Accuracy","Offline","Langs"),
        ("Gemma 4 alone","16.7%","✅","Any"),
        ("CNN alone","94.0%","✅","❌"),
        ("CropDoc v12","93.0%","✅","❌"),
        ("CropDoc v26 ★","99.3%","✅","10"),
    ]
    cws=[300,140,120,120]
    cxs=[55]; [cxs.append(cxs[-1]+c) for c in cws[:-1]]

    for ri,row in enumerate(rows):
        rp3=ease_out(min((t-2.0-ri*0.25)/1.0,1.0))
        if rp3<=0: continue
        ra3=int(255*rp3)
        is_h=(ri==0); is_b=(ri==4)
        if is_b:
            hl2=Image.new("RGBA",img.size,(0,0,0,0))
            hld=ImageDraw.Draw(hl2)
            hld.rounded_rectangle([50,ty3-2,W-50,ty3+26],radius=6,fill=(0,80,40,80))
            img=Image.alpha_composite(img.convert("RGBA"),hl2).convert("RGB")
            draw=ImageDraw.Draw(img)
        for ci,(cell,cx3) in enumerate(zip(row,cxs)):
            fc3=GOLD if (is_b and ci==1) else (ACCENT if is_h else (NEON if is_b else WHITE))
            draw.text((cx3+6,ty3+2),cell,font=fnt(FONT_BOLD if is_h or is_b else FONT_REG,16),fill=(*fc3,ra3))
        draw.line([(55,ty3+28),(W-55,ty3+28)],fill=(30,70,45),width=1)
        ty3+=30

    return np.array(img)

# ════════════════════════════════════════════════════
# SCENE 6: IMPACT + CTA (125~150s)
# ════════════════════════════════════════════════════
def scene_impact(t):
    pulse=0.5+0.5*math.sin(t*2.5)
    img=make_bg()
    img=add_glow(img,W//2,H//2,500,ACCENT,45)
    img=add_glow(img,W//2,H//2,200,NEON,int(20*pulse))
    draw=ImageDraw.Draw(img)

    draw.text((W//2-260,30),"REAL-WORLD IMPACT",font=fnt(FONT_MONO,18),fill=(*ACCENT,220))
    draw.text((W//2-430,65),"CropDoc: AI for the Farmers Who Need It Most",font=fnt(FONT_BOLD,34),fill=WHITE)
    draw.line([(55,110),(W-55,110)],fill=(40,100,60),width=1)

    data=[
        ("500M","Farmers\nServed",          ACCENT,"🌱"),
        ("72",  "Disease\nClasses",          TEAL,  "🔬"),
        ("$220B","Losses\nPrevented",        GOLD,  "💰"),
        ("10",  "Languages\nSupported",      PURPLE,"🌍"),
        ("<30s","Diagnosis\nTime",           NEON,  "⚡"),
        ("0",   "Internet\nRequired",        ORANGE,"📡"),
    ]
    iw=(W-170)//3; ih=185
    ix,iy2=55,125
    for idx,(val,desc,col,icon) in enumerate(data):
        show=ease_out(min((t-idx*0.3)/1.5,1.0))
        if show<=0: continue
        a=int(255*show)
        ci=idx%3; ri=idx//3
        bx=ix+ci*(iw+25); by=iy2+ri*(ih+18)
        img=draw_panel(img,bx,by,bx+iw,by+ih,fill=(col[0]//7,col[1]//7,col[2]//7,200),col=col,radius=14)
        draw=ImageDraw.Draw(img)
        draw.text((bx+16,by+14),icon,font=fnt(FONT_BOLD,36),fill=(*col,a))
        draw.text((bx+70,by+16),val,font=fnt(FONT_BOLD,40),fill=(*col,a))
        for di2,dl2 in enumerate(desc.split('\n')):
            draw.text((bx+16,by+65+di2*22),dl2,font=fnt(FONT_REG,16),fill=(*GREY,a))

    # CTA
    cta_y=560
    draw.line([(55,cta_y),(W-55,cta_y)],fill=(40,100,60),width=1); cta_y+=16
    cta_a=int(180+75*pulse)
    cta_l=Image.new("RGBA",img.size,(0,0,0,0))
    ctad=ImageDraw.Draw(cta_l)
    ctad.rounded_rectangle([W//2-280,cta_y,W//2+280,cta_y+60],radius=30,
        fill=(0,150,75,200),outline=(*NEON,cta_a),width=3)
    img=Image.alpha_composite(img.convert("RGBA"),cta_l).convert("RGB")
    draw=ImageDraw.Draw(img)
    draw.text((W//2-220,cta_y+15),"🚀  Try CropDoc — Free & Open Source",font=fnt(FONT_BOLD,24),fill=WHITE)

    draw.text((W//2-370,cta_y+75),"🤗  huggingface.co/spaces/noivan/cropdoc",font=fnt(FONT_BOLD,17),fill=(100,200,255))
    draw.text((W//2+30, cta_y+75),"💻  github.com/noivan0/cropdoc",font=fnt(FONT_BOLD,17),fill=(180,210,180))
    draw.text((W//2-330,cta_y+105),"Gemma 4 E4B-IT · CC-BY 4.0 · Google DeepMind Gemma 4 Good Hackathon 2026",
              font=fnt(FONT_REG,14),fill=GREY)

    return np.array(img)

# ════════════════════════════════════════════════════
# RENDER
# ════════════════════════════════════════════════════
FPS=12
FADE=0.6

scenes=[
    ("title",   scene_title,   10.0),
    ("problem", scene_problem, 18.0),
    ("solution",scene_solution,20.0),
    ("demo",    scene_demo,    47.0),
    ("tech",    scene_tech,    30.0),
    ("impact",  scene_impact,  25.0),
]

total=sum(d for _,_,d in scenes)
total_f=int(total*FPS)
print(f"총 {total:.0f}초 ({total/60:.1f}분) / {total_f}프레임 / {W}x{H}@{FPS}fps")

OUT="/tmp/cropdoc_demo.mp4"
writer=imageio.get_writer(OUT,fps=FPS,codec='libx264',quality=8,macro_block_size=1,
    ffmpeg_params=['-crf','22','-preset','fast','-pix_fmt','yuv420p'],
    ffmpeg_log_level='quiet')

prev_last=None
for sname,sfn,dur in scenes:
    nf=int(dur*FPS)
    print(f"  {sname}: {nf}프레임...", flush=True)
    last=None
    for fi in range(nf):
        t=fi/FPS
        frame=sfn(t)
        # 페이드인
        if fi<int(FADE*FPS) and prev_last is not None:
            alpha=fi/(FADE*FPS)
            frame=(prev_last*(1-alpha)+frame*alpha).astype(np.uint8)
        writer.append_data(frame)
        last=frame
    prev_last=last

writer.close()
sz=os.path.getsize(OUT)
print(f"\n✅ {OUT}  ({sz/1024/1024:.1f}MB)  {total:.0f}초")
