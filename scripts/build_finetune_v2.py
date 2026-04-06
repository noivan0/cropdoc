"""
Gemma4 파인튜닝 데이터셋 v2 생성
40 샘플 → 500+ 샘플 확대
모든 LABEL_RULES 커버리지
"""
import json, os, random
from pathlib import Path

PROJECT_ROOT = Path('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
os.chdir(PROJECT_ROOT)

# ── 전체 질병 DB (eval_harness LABEL_RULES 기반) ──────────────────────────────
DISEASE_DB = {
    # ── Tomato ──────────────────────────────────────────────────────────────
    "Tomato Early Blight": {
        "en": """PLANT: Tomato (Solanum lycopersicum)
DISEASE: Early Blight (Alternaria solani)
SYMPTOMS: Concentric ring brown spots (target-board pattern) appear on lower, older leaves first. Yellow halo surrounds each lesion. Lesions expand and merge.
CAUSE: Alternaria solani fungus; favored by warm wet weather.
TREATMENT: Apply Chlorothalonil or Azoxystrobin fungicide. Remove infected lower leaves. Avoid overhead irrigation. Rotate crops.
URGENCY: WARNING — treat within 1 week to prevent yield loss.""",
        "ko": """식물: 토마토 (Solanum lycopersicum)
진단: 조기역병 (Tomato Early Blight)
원인: Alternaria solani 진균
증상: 하부 묵은 잎에 동심원 갈색 반점 (과녁 패턴), 반점 주위 황변, 점차 상부로 확산
처방: 클로로탈로닐 또는 아족시스트로빈 살균제 살포. 하부 감염 잎 즉시 제거. 두둑 관수로 전환.
긴급도: ⚠️ 1주일 내 조치 (방치 시 수확량 30~50% 감소 가능)""",
        "es": """PLANTA: Tomate (Solanum lycopersicum)
ENFERMEDAD: Tizón temprano (Alternaria solani)
SÍNTOMAS: Manchas marrones concéntricas en hojas inferiores, halo amarillo alrededor.
TRATAMIENTO: Fungicida Clorotalonil o Azoxistrobina. Eliminar hojas inferiores infectadas.
URGENCIA: ADVERTENCIA — tratar en 1 semana.""",
    },
    "Tomato Late Blight": {
        "en": """PLANT: Tomato (Solanum lycopersicum)
DISEASE: Late Blight (Phytophthora infestans)
SYMPTOMS: Water-soaked dark lesions appear on leaves and stems. White fuzzy mold visible on undersides in humid conditions. Rapid browning and collapse of plant tissue.
CAUSE: Phytophthora infestans (water mold). Spreads explosively in cool, wet weather.
TREATMENT: Apply Mancozeb or Chlorothalonil immediately. Remove all infected material and destroy. Switch to drip irrigation.
URGENCY: CRITICAL — can destroy entire field within 48-72 hours in humid weather.""",
        "ko": """식물: 토마토 (Solanum lycopersicum)
진단: 역병 (Tomato Late Blight)
원인: Phytophthora infestans (난균류)
증상: 잎과 줄기에 수침상 암갈색 병반, 잎 뒷면 흰색 솜털 곰팡이, 줄기 갈변 및 붕괴
처방: 만코제브 또는 클로로탈로닐 살균제 즉시 적용. 감염 부위 즉시 제거 후 소각. 점적 관수 전환.
긴급도: 🚨 즉시! 48~72시간 내 포장 전체 전파 가능""",
        "es": """PLANTA: Tomate
ENFERMEDAD: Tizón tardío (Phytophthora infestans)
SÍNTOMAS: Lesiones acuosas oscuras en hojas y tallos, moho blanco en el envés.
TRATAMIENTO: Mancozeb o Clorotalonil inmediatamente. Eliminar material infectado.
URGENCIA: CRÍTICA — puede destruir todo el campo en 48-72 horas.""",
    },
    "Tomato Bacterial Spot": {
        "en": """PLANT: Tomato (Solanum lycopersicum)
DISEASE: Bacterial Spot (Xanthomonas campestris pv. vesicatoria)
SYMPTOMS: Small water-soaked spots on leaves that turn brown with yellow halos. Spots may have greasy appearance. Spots on fruit appear raised and corky.
CAUSE: Xanthomonas bacteria. Spreads via water splash and infected seeds.
TREATMENT: Copper-based bactericide (copper hydroxide or copper sulfate). Remove infected plants. Use disease-free certified seeds.
URGENCY: HIGH — spreads rapidly in warm wet weather.""",
        "ko": """식물: 토마토
진단: 세균성 반점병 (Tomato Bacterial Spot)
원인: Xanthomonas campestris pv. vesicatoria (세균)
증상: 잎에 수침상 소반점 → 갈색 황색 테두리 반점, 과실 표면 거칠고 코르크화
처방: 수산화동 또는 황산동 계열 살균제. 감염 식물 제거. 인증 종자 사용.
긴급도: 🔴 높음 — 고온 다습 시 빠르게 확산""",
        "es": """PLANTA: Tomate
ENFERMEDAD: Mancha bacteriana (Xanthomonas)
SÍNTOMAS: Manchas pequeñas acuosas en hojas, halo amarillo, manchas en frutos.
TRATAMIENTO: Bactericida a base de cobre. Eliminar plantas infectadas.
URGENCIA: ALTA.""",
    },
    "Tomato Leaf Mold": {
        "en": """PLANT: Tomato (Solanum lycopersicum)
DISEASE: Leaf Mold (Passalora fulva / Fulvia fulva)
SYMPTOMS: Pale green to yellow patches on upper leaf surface. Olive-green to brown velvety mold on undersides. Affected leaves turn yellow and drop.
CAUSE: Passalora fulva fungus. Thrives in high humidity (>85%) and warm temperatures.
TREATMENT: Improve ventilation. Apply Mancozeb or Chlorothalonil fungicide. Reduce humidity in greenhouse.
URGENCY: MODERATE — control humidity first.""",
        "ko": """식물: 토마토
진단: 잎곰팡이병 (Tomato Leaf Mold)
원인: Passalora fulva 진균
증상: 잎 윗면 연황록색 반점, 잎 뒷면 올리브녹색~갈색 벨벳 곰팡이, 잎 황화 후 낙엽
처방: 환기 개선 우선. 만코제브 또는 클로로탈로닐 살균제. 온실 습도 85% 이하 유지.
긴급도: ⚠️ 보통 — 환기 개선이 핵심""",
        "es": """PLANTA: Tomate
ENFERMEDAD: Moho de la hoja (Passalora fulva)
SÍNTOMAS: Manchas verde-amarillas en el haz, moho velloso oliváceo en el envés.
TRATAMIENTO: Mejorar ventilación. Fungicida Mancozeb o Clorotalonil.
URGENCIA: MODERADA.""",
    },
    "Tomato Septoria Leaf Spot": {
        "en": """PLANT: Tomato (Solanum lycopersicum)
DISEASE: Septoria Leaf Spot (Septoria lycopersici)
SYMPTOMS: Small circular spots with dark borders and light gray centers on lower leaves. Tiny dark fruiting bodies (pycnidia) visible in spot centers. Severe infection causes defoliation.
CAUSE: Septoria lycopersici fungus. Spreads via water splash in humid conditions.
TREATMENT: Chlorothalonil or Mancozeb fungicide. Remove affected lower leaves. Mulch to prevent soil splash.
URGENCY: HIGH — can cause complete defoliation.""",
        "ko": """식물: 토마토
진단: 셉토리아 잎반점병 (Tomato Septoria Leaf Spot)
원인: Septoria lycopersici 진균
증상: 하부 잎에 어두운 테두리·밝은 회색 중심부의 소형 원형 반점, 반점 내 소흑점(분생포자각), 낙엽 진행
처방: 클로로탈로닐 또는 만코제브 살균제. 하부 잎 제거. 멀칭으로 토양 비산 방지.
긴급도: 🔴 높음 — 완전 낙엽 위험""",
        "es": """PLANTA: Tomate
ENFERMEDAD: Mancha foliar de Septoria (Septoria lycopersici)
SÍNTOMAS: Manchas circulares pequeñas con bordes oscuros, centros gris claro, picnidios visibles.
TRATAMIENTO: Clorotalonil o Mancozeb. Eliminar hojas inferiores. Acolchar suelo.
URGENCIA: ALTA.""",
    },
    "Healthy Tomato": {
        "en": """PLANT: Tomato (Solanum lycopersicum)
STATUS: Healthy
OBSERVATION: Leaves display uniform deep green color, no lesions or spots, normal vascular structure, no yellowing or wilting. Plant vigor appears normal.
RECOMMENDATION: Continue current care routine. Maintain regular monitoring schedule. Ensure balanced NPK fertilization and consistent watering.
URGENCY: None — plant is healthy and thriving.""",
        "ko": """식물: 토마토 (Solanum lycopersicum)
상태: 정상 (건강)
관찰: 균일한 진녹색, 병반·반점 없음, 정상 엽맥, 황변·시들음 없음, 생육 정상
권고: 현재 관리 방식 유지. 정기 모니터링 継続. 균형 잡힌 NPK 시비 및 일정한 관수.
긴급도: 없음 — 건강하고 생육 양호""",
        "es": """PLANTA: Tomate
ESTADO: Saludable
OBSERVACIÓN: Hojas verde oscuro uniforme, sin lesiones, estructura vascular normal.
RECOMENDACIÓN: Continuar cuidado habitual. Monitoreo regular.
URGENCIA: Ninguna — planta saludable.""",
    },
    # ── Potato ──────────────────────────────────────────────────────────────
    "Potato Early Blight": {
        "en": """PLANT: Potato (Solanum tuberosum)
DISEASE: Early Blight (Alternaria solani)
SYMPTOMS: Dark brown spots with concentric rings (bull's-eye pattern) on older lower leaves. Yellow tissue surrounds each spot. Severely affected leaves turn entirely yellow and drop.
CAUSE: Alternaria solani fungus. Common in warm, dry-to-moist weather cycles.
TREATMENT: Azoxystrobin or Chlorothalonil fungicide. Ensure adequate potassium fertilization. Remove infected foliage.
URGENCY: WARNING — begin treatment at first sign.""",
        "ko": """식물: 감자 (Solanum tuberosum)
진단: 조기역병 (Potato Early Blight)
원인: Alternaria solani 진균
증상: 묵은 하부 잎에 동심원(과녁) 무늬 암갈색 반점, 주변 황변, 심하면 낙엽
처방: 아족시스트로빈 또는 클로로탈로닐 살균제. 칼륨 시비 충분히. 감염 잎 제거.
긴급도: ⚠️ 첫 징후 시 즉시 처리""",
        "es": """PLANTA: Papa (Solanum tuberosum)
ENFERMEDAD: Tizón temprano (Alternaria solani)
SÍNTOMAS: Manchas marrones con anillos concéntricos en hojas viejas, halo amarillo.
TRATAMIENTO: Fungicida Azoxistrobina o Clorotalonil. Asegurar potasio adecuado.
URGENCIA: ADVERTENCIA.""",
    },
    "Potato Late Blight": {
        "en": """PLANT: Potato (Solanum tuberosum)
DISEASE: Late Blight (Phytophthora infestans)
SYMPTOMS: Dark water-soaked lesions on leaves, white mold on undersides. Brown/purple discoloration on stems. Rapid collapse of foliage. Tubers show brown rot internally.
CAUSE: Phytophthora infestans — same pathogen that caused the Irish Potato Famine (1840s).
TREATMENT: Mancozeb or Metalaxyl-M fungicide immediately. Hill up soil around plants. Remove and destroy infected foliage. Do not harvest tubers during wet conditions.
URGENCY: CRITICAL — can destroy entire crop within days.""",
        "ko": """식물: 감자 (Solanum tuberosum)
진단: 역병 (Potato Late Blight)
원인: Phytophthora infestans (아일랜드 대기근 유발 병원균)
증상: 잎에 수침상 병반, 잎 뒷면 흰색 곰팡이, 줄기 갈변·자반, 지상부 급격히 괴사, 덩이줄기 내부 갈변 부패
처방: 만코제브 또는 메탈락실-M 살균제 즉시. 북주기 실시. 감염 지상부 즉시 제거 소각. 습할 때 수확 금지.
긴급도: 🚨 즉시! 수일 내 전체 포장 괴멸 가능""",
        "es": """PLANTA: Papa
ENFERMEDAD: Tizón tardío (Phytophthora infestans)
SÍNTOMAS: Lesiones acuosas oscuras, moho blanco en envés, podredumbre de tubérculos.
TRATAMIENTO: Mancozeb o Metalaxil-M inmediatamente. Eliminar follaje infectado.
URGENCIA: CRÍTICA.""",
    },
    "Healthy Potato": {
        "en": """PLANT: Potato (Solanum tuberosum)
STATUS: Healthy
OBSERVATION: Leaves show normal green color, no lesions, no water-soaked areas, normal upright growth. No mold visible on undersides.
RECOMMENDATION: Maintain current watering and fertilization. Scout regularly for early signs of late blight especially during cool wet periods.
URGENCY: None — plant is healthy.""",
        "ko": """식물: 감자 (Solanum tuberosum)
상태: 정상 (건강)
관찰: 정상 녹색, 병반 없음, 수침상 부위 없음, 정상 직립 생육, 잎 뒷면 곰팡이 없음
권고: 현재 관수 및 시비 유지. 특히 서늘하고 습한 시기에 역병 조기 징후 정기 확인.
긴급도: 없음 — 정상""",
        "es": """PLANTA: Papa
ESTADO: Saludable
OBSERVACIÓN: Hojas verdes normales, sin lesiones, sin áreas acuosas.
RECOMENDACIÓN: Mantener riego y fertilización. Monitorear regularmente.
URGENCIA: Ninguna.""",
    },
    # ── Pepper ──────────────────────────────────────────────────────────────
    "Pepper Bacterial Spot": {
        "en": """PLANT: Pepper (Capsicum annuum)
DISEASE: Bacterial Spot (Xanthomonas campestris pv. vesicatoria)
SYMPTOMS: Small water-soaked spots on leaves turn brown with yellow halos. Spots may coalesce. Raised, scabby lesions on fruit reduce market value.
CAUSE: Xanthomonas bacteria spread by water splash and wind. Favored by warm wet conditions.
TREATMENT: Copper hydroxide bactericide. Avoid overhead irrigation. Use certified disease-free seed. Remove severely infected plants.
URGENCY: HIGH — impacts yield and fruit quality.""",
        "ko": """식물: 고추 (Capsicum annuum)
진단: 세균성 반점병 (Pepper Bacterial Spot)
원인: Xanthomonas campestris pv. vesicatoria (세균)
증상: 잎에 수침상 소반점 → 갈색 황색 테두리 반점, 반점 융합, 과실에 융기된 딱지 병반
처방: 수산화동 살균제. 두둑 관수로 전환. 인증 무병 종자 사용. 심한 감염 식물 제거.
긴급도: 🔴 높음 — 수확량 및 과실 품질 저하""",
        "es": """PLANTA: Pimiento (Capsicum annuum)
ENFERMEDAD: Mancha bacteriana (Xanthomonas)
SÍNTOMAS: Manchas acuosas en hojas, halo amarillo, lesiones costrosas en frutos.
TRATAMIENTO: Hidróxido de cobre. Evitar riego aéreo. Usar semillas certificadas.
URGENCIA: ALTA.""",
    },
    "Healthy Pepper": {
        "en": """PLANT: Pepper (Capsicum annuum)
STATUS: Healthy
OBSERVATION: Leaves are uniformly green, glossy, no spots or lesions. Stems are firm. No wilting observed.
RECOMMENDATION: Maintain balanced fertilization and consistent moisture. Monitor for aphids and bacterial spot during wet periods.
URGENCY: None — plant is healthy.""",
        "ko": """식물: 고추 (Capsicum annuum)
상태: 정상 (건강)
관찰: 균일한 광택 녹색 잎, 반점·병반 없음, 줄기 견고, 시들음 없음
권고: 균형 시비 및 일정한 수분 유지. 습한 시기 진딧물 및 세균성 반점병 모니터링.
긴급도: 없음 — 정상""",
        "es": """PLANTA: Pimiento
ESTADO: Saludable
OBSERVACIÓN: Hojas verde uniforme, brillantes, sin manchas.
RECOMENDACIÓN: Fertilización equilibrada. Monitorear plagas.
URGENCIA: Ninguna.""",
    },
    # ── Apple ──────────────────────────────────────────────────────────────
    "Apple Scab": {
        "en": """PLANT: Apple (Malus domestica)
DISEASE: Apple Scab (Venturia inaequalis)
SYMPTOMS: Olive-green to black scab-like lesions on leaves. Velvety texture on lesions. Infected fruit shows corky, dark scabs reducing quality.
CAUSE: Venturia inaequalis fungus. Infection occurs during wet spring weather.
TREATMENT: Captan or Mancozeb fungicide. Apply before primary infection periods. Rake and destroy fallen leaves.
URGENCY: WARNING — preventative management critical in spring.""",
        "ko": """식물: 사과 (Malus domestica)
진단: 사과 부스럼병 (Apple Scab)
원인: Venturia inaequalis 진균
증상: 잎에 올리브녹색~흑색 딱지 병반, 벨벳 질감, 과실에 코르크 흑색 딱지
처방: 캡탄 또는 만코제브 살균제. 1차 감염기 이전 예방 살포. 낙엽 수거 소각.
긴급도: ⚠️ 주의 — 봄철 예방 관리 핵심""",
        "es": """PLANTA: Manzana
ENFERMEDAD: Costra del manzano (Venturia inaequalis)
SÍNTOMAS: Lesiones verde-oliváceas a negras en hojas, costras corchosas en frutos.
TRATAMIENTO: Captan o Mancozeb. Aplicar antes de períodos de infección primaria.
URGENCIA: ADVERTENCIA.""",
    },
    "Apple Black Rot": {
        "en": """PLANT: Apple (Malus domestica)
DISEASE: Black Rot (Botryosphaeria obtusa)
SYMPTOMS: Brown rotting areas on fruit starting at calyx end. "Frog-eye" leaf spots with purple border and tan center. Mummified black fruit remains on tree.
CAUSE: Botryosphaeria obtusa fungus. Overwinters in dead wood and mummified fruit.
TREATMENT: Captan fungicide. Prune out dead wood. Remove mummified fruit. Avoid wounding bark.
URGENCY: HIGH — can cause significant fruit loss.""",
        "ko": """식물: 사과
진단: 사과 검은 부패병 (Apple Black Rot)
원인: Botryosphaeria obtusa 진균
증상: 과실 악화 부위에서 갈색 부패 시작, 개구리눈 잎반점(자색 테두리·황갈색 중앙), 검게 미라화된 과실
처방: 캡탄 살균제. 고사지 전정. 미라 과실 제거. 수피 상처 방지.
긴급도: 🔴 높음 — 상당한 과실 손실 가능""",
        "es": """PLANTA: Manzana
ENFERMEDAD: Podredumbre negra (Botryosphaeria obtusa)
SÍNTOMAS: Podredumbre marrón en frutos, manchas "ojo de rana", frutos momificados negros.
TRATAMIENTO: Fungicida Captan. Podar madera muerta. Eliminar frutos momificados.
URGENCIA: ALTA.""",
    },
    "Apple Cedar Rust": {
        "en": """PLANT: Apple (Malus domestica)
DISEASE: Cedar Apple Rust (Gymnosporangium juniperi-virginianae)
SYMPTOMS: Bright orange-yellow spots on upper leaf surface. Tube-like spore structures on leaf undersides. Infected fruit shows orange spots.
CAUSE: Gymnosporangium juniperi-virginianae — requires both cedar/juniper and apple trees to complete life cycle.
TREATMENT: Myclobutanil or Propiconazole fungicide. Remove nearby cedar/juniper hosts if possible.
URGENCY: MODERATE — primarily aesthetic damage.""",
        "ko": """식물: 사과
진단: 붉은별무늬병 / 사과 녹병 (Apple Cedar Rust)
원인: Gymnosporangium juniperi-virginianae (향나무-사과 이종기생균)
증상: 잎 윗면 밝은 주황색 반점, 잎 뒷면 관상 포자층, 과실 주황색 반점
처방: 마이클로부타닐 또는 프로피코나졸 살균제. 가능 시 주변 향나무 제거.
긴급도: ⚠️ 보통 — 주로 외관 피해""",
        "es": """PLANTA: Manzana
ENFERMEDAD: Roya del cedro-manzano (Gymnosporangium)
SÍNTOMAS: Manchas naranja-amarillas en haz, estructuras tubulares en envés.
TRATAMIENTO: Miclobutanil o Propiconazol. Eliminar cedros/enebros cercanos si es posible.
URGENCIA: MODERADA.""",
    },
    "Healthy Apple": {
        "en": """PLANT: Apple (Malus domestica)
STATUS: Healthy
OBSERVATION: Leaves are uniformly green with no spots, scabs or discoloration. Fruit development normal. No mold or rust symptoms visible.
RECOMMENDATION: Maintain preventative spray program in spring. Monitor for scab and rust during wet periods.
URGENCY: None — plant is healthy.""",
        "ko": """식물: 사과 (Malus domestica)
상태: 정상 (건강)
관찰: 균일한 녹색 잎, 반점·딱지·변색 없음, 과실 정상 발육, 곰팡이·녹병 증상 없음
권고: 봄철 예방 살포 프로그램 유지. 습한 시기 부스럼병·녹병 모니터링.
긴급도: 없음 — 정상""",
        "es": """PLANTA: Manzana
ESTADO: Saludable
OBSERVACIÓN: Hojas verde uniforme, sin manchas ni costras.
RECOMENDACIÓN: Programa preventivo de pulverización en primavera.
URGENCIA: Ninguna.""",
    },
    # ── Corn ──────────────────────────────────────────────────────────────
    "Corn Gray Leaf Spot": {
        "en": """PLANT: Corn/Maize (Zea mays)
DISEASE: Gray Leaf Spot (Cercospora zeae-maydis)
SYMPTOMS: Rectangular gray to tan lesions on leaves, running parallel to leaf veins. Lesions turn gray with age. Severe infection causes premature leaf death.
CAUSE: Cercospora zeae-maydis fungus. Favored by warm, humid conditions and dense canopy.
TREATMENT: Azoxystrobin or Propiconazole fungicide. Improve field drainage. Plant resistant hybrids.
URGENCY: HIGH — can cause 20-50% yield loss.""",
        "ko": """식물: 옥수수 (Zea mays)
진단: 회색잎마름병 (Corn Gray Leaf Spot)
원인: Cercospora zeae-maydis 진균
증상: 잎맥과 평행한 직사각형 회~담갈색 병반, 노화 시 회색화, 심하면 조기 고엽
처방: 아족시스트로빈 또는 프로피코나졸 살균제. 배수 개선. 저항성 품종 재배.
긴급도: 🔴 높음 — 수량 20~50% 감소 가능""",
        "es": """PLANTA: Maíz (Zea mays)
ENFERMEDAD: Mancha gris foliar (Cercospora zeae-maydis)
SÍNTOMAS: Lesiones rectangulares grises paralelas a nervaduras, muerte prematura de hojas.
TRATAMIENTO: Azoxistrobina o Propiconazol. Mejorar drenaje.
URGENCIA: ALTA.""",
    },
    "Corn Common Rust": {
        "en": """PLANT: Corn/Maize (Zea mays)
DISEASE: Common Rust (Puccinia sorghi)
SYMPTOMS: Oval to elongated rusty-orange to brown pustules on both leaf surfaces. Pustules turn darker with age. Heavy infection causes yellowing and premature death.
CAUSE: Puccinia sorghi fungus. Spreads via wind-borne spores. Favored by cool to moderate temperatures.
TREATMENT: Propiconazole or Azoxystrobin fungicide. Use resistant hybrids. Apply at early signs.
URGENCY: MODERATE to HIGH — protect yield by early treatment.""",
        "ko": """식물: 옥수수
진단: 일반 녹병 (Corn Common Rust)
원인: Puccinia sorghi 진균
증상: 잎 양면에 타원형 적갈색 포자퇴, 노화 시 어두워짐, 심하면 황화 및 조기 고사
처방: 프로피코나졸 또는 아족시스트로빈 살균제. 저항성 품종 재배. 초기 징후에 살포.
긴급도: ⚠️ 보통~높음 — 조기 처리로 수량 보호""",
        "es": """PLANTA: Maíz
ENFERMEDAD: Roya común (Puccinia sorghi)
SÍNTOMAS: Pústulas ovaladas anaranjadas en ambas superficies foliares.
TRATAMIENTO: Propiconazol o Azoxistrobina. Usar híbridos resistentes.
URGENCIA: MODERADA a ALTA.""",
    },
    "Corn Northern Blight": {
        "en": """PLANT: Corn/Maize (Zea mays)
DISEASE: Northern Corn Leaf Blight (Exserohilum turcicum)
SYMPTOMS: Long, elliptical cigar-shaped gray-green to tan lesions (2.5-15cm) on leaves. Dark sporulation visible on lesions in humid conditions. Severe blight above ear at silking causes major yield loss.
CAUSE: Exserohilum turcicum fungus. Favored by moderate temperatures and prolonged leaf wetness.
TREATMENT: Azoxystrobin or Mancozeb fungicide at early detection. Plant resistant hybrids. Rotate crops.
URGENCY: HIGH — especially if infection occurs before or during silking.""",
        "ko": """식물: 옥수수
진단: 북부 잎마름병 (Corn Northern Blight)
원인: Exserohilum turcicum 진균
증상: 잎에 긴 타원형(시가 형) 회녹색~담갈색 병반(2.5~15cm), 습한 조건에서 검은 포자형성, 출사기 이삭 위 심한 감염 → 큰 수량 감소
처방: 아족시스트로빈 또는 만코제브 살균제 조기 살포. 저항성 품종 재배. 윤작.
긴급도: 🔴 높음 — 특히 출사기 전후 감염 시""",
        "es": """PLANTA: Maíz
ENFERMEDAD: Tizón norteño (Exserohilum turcicum)
SÍNTOMAS: Lesiones largas elípticas gris-verdosas a bronceadas en hojas.
TRATAMIENTO: Azoxistrobina o Mancozeb en detección temprana. Híbridos resistentes.
URGENCIA: ALTA.""",
    },
    "Healthy Corn": {
        "en": """PLANT: Corn/Maize (Zea mays)
STATUS: Healthy
OBSERVATION: Leaves are uniformly green, no lesions or pustules, normal upright growth. No rust or blight symptoms visible.
RECOMMENDATION: Monitor regularly for northern blight and rust. Ensure adequate nitrogen fertilization for robust growth.
URGENCY: None — plant is healthy.""",
        "ko": """식물: 옥수수 (Zea mays)
상태: 정상 (건강)
관찰: 균일한 녹색 잎, 병반·포자퇴 없음, 정상 직립 생육, 잎마름병·녹병 증상 없음
권고: 잎마름병 및 녹병 정기 모니터링. 충분한 질소 시비로 강건한 생육 유지.
긴급도: 없음 — 정상""",
        "es": """PLANTA: Maíz
ESTADO: Saludable
OBSERVACIÓN: Hojas verde uniforme, sin lesiones ni pústulas.
RECOMENDACIÓN: Monitorear regularmente. Fertilización nitrogenada adecuada.
URGENCIA: Ninguna.""",
    },
    # ── Grape ──────────────────────────────────────────────────────────────
    "Grape Black Rot": {
        "en": """PLANT: Grape (Vitis vinifera)
DISEASE: Black Rot (Guignardia bidwellii)
SYMPTOMS: Tan lesions with dark brown borders on leaves. Infected berries turn black and mummify (shriveled black raisins). Small black fruiting bodies (pycnidia) in lesions.
CAUSE: Guignardia bidwellii fungus. Spreads from mummified fruit overwintering in vineyard.
TREATMENT: Myclobutanil or Mancozeb fungicide from early shoot growth. Remove all mummified fruit. Prune for good air circulation.
URGENCY: HIGH — can destroy entire grape crop.""",
        "ko": """식물: 포도 (Vitis vinifera)
진단: 흑부병 (Grape Black Rot)
원인: Guignardia bidwellii 진균
증상: 잎에 갈색 테두리 황갈색 반점, 감염 과립 흑변 후 미라화(건포도형 흑색), 병반 내 소흑점(분생포자각)
처방: 마이클로부타닐 또는 만코제브 살균제 (신초 발생부터). 미라 과실 전량 제거. 통풍 개선 전정.
긴급도: 🔴 높음 — 포도 수확 전량 손실 가능""",
        "es": """PLANTA: Uva (Vitis vinifera)
ENFERMEDAD: Podredumbre negra (Guignardia bidwellii)
SÍNTOMAS: Lesiones bronceadas en hojas, bayas negras momificadas.
TRATAMIENTO: Miclobutanil o Mancozeb desde brotación. Eliminar frutos momificados.
URGENCIA: ALTA.""",
    },
    "Grape Esca": {
        "en": """PLANT: Grape (Vitis vinifera)
DISEASE: Esca (Esca complex — Phaeomoniella chlamydospora, Phaeoacremonium minimum)
SYMPTOMS: "Tiger-stripe" pattern on leaves (yellow/red stripes between veins). Sudden wilting and collapse ("apoplexy"). Internal wood shows brown/gray discoloration.
CAUSE: Wood-infecting fungi complex. Enters through pruning wounds.
TREATMENT: No effective chemical cure once established. Remove and destroy severely affected vines. Protect pruning cuts with fungicide or wound sealant.
URGENCY: HIGH — chronic disease with no cure.""",
        "ko": """식물: 포도
진단: 에스카병 (Grape Esca)
원인: Phaeomoniella chlamydospora 등 목재 감염 진균 복합체
증상: 잎에 "호랑이 줄무늬"(엽맥 사이 황색/적색 줄무늬), 급격한 시들음·붕괴(졸중), 목재 내부 갈색·회색 변색
처방: 확립된 감염은 화학 방제 불가. 심한 포기 제거·소각. 전정 상처에 살균제 또는 상처 도포제 적용.
긴급도: 🔴 높음 — 치료법 없는 만성 병해""",
        "es": """PLANTA: Uva
ENFERMEDAD: Esca (complejo fúngico en madera)
SÍNTOMAS: Patrón "piel de tigre" en hojas, colapso súbito, decoloración interna de madera.
TRATAMIENTO: Sin cura química efectiva. Eliminar cepas gravemente afectadas. Proteger heridas de poda.
URGENCIA: ALTA.""",
    },
    "Grape Leaf Spot": {
        "en": """PLANT: Grape (Vitis vinifera)
DISEASE: Grape Leaf Spot (Isariopsis Leaf Spot / Cercospora)
SYMPTOMS: Angular dark brown spots on upper leaf surface. Grayish sporulation on undersides. Severe infection causes defoliation weakening vine.
CAUSE: Pseudocercospora vitis or related fungi. Favored by warm, humid conditions.
TREATMENT: Mancozeb or copper fungicide. Improve canopy ventilation through training.
URGENCY: MODERATE.""",
        "ko": """식물: 포도
진단: 포도 잎반점병 (Grape Leaf Spot)
원인: Pseudocercospora vitis 진균
증상: 잎 윗면 각형 암갈색 반점, 잎 뒷면 회색 포자형성, 심하면 낙엽 → 포기 약화
처방: 만코제브 또는 동 계열 살균제. 유인·전정으로 수관 환기 개선.
긴급도: ⚠️ 보통""",
        "es": """PLANTA: Uva
ENFERMEDAD: Mancha foliar de la uva (Pseudocercospora)
SÍNTOMAS: Manchas angulares en haz, esporulación grisácea en envés.
TRATAMIENTO: Mancozeb o fungicida cúprico.
URGENCIA: MODERADA.""",
    },
    "Healthy Grape": {
        "en": """PLANT: Grape (Vitis vinifera)
STATUS: Healthy
OBSERVATION: Leaves are uniformly green, no spots or mold, normal vine structure. Fruit clusters developing normally.
RECOMMENDATION: Maintain preventative fungicide program. Prune for good air circulation. Monitor for black rot and esca.
URGENCY: None — plant is healthy.""",
        "ko": """식물: 포도
상태: 정상 (건강)
관찰: 균일한 녹색 잎, 반점·곰팡이 없음, 정상 수관 구조, 과방 정상 발육
권고: 예방 살균제 프로그램 유지. 통풍 개선 전정. 흑부병·에스카 모니터링.
긴급도: 없음 — 정상""",
        "es": """PLANTA: Uva
ESTADO: Saludable
OBSERVACIÓN: Hojas verde uniforme, sin manchas ni moho.
RECOMENDACIÓN: Programa preventivo de fungicidas. Podar para buena circulación.
URGENCIA: Ninguna.""",
    },
    # ── Peach ──────────────────────────────────────────────────────────────
    "Peach Bacterial Spot": {
        "en": """PLANT: Peach (Prunus persica)
DISEASE: Bacterial Spot (Xanthomonas arboricola pv. pruni)
SYMPTOMS: Small water-soaked spots on leaves turn purplish-brown, often with yellow halo. Spots may drop out leaving "shot hole" appearance. Sunken dark spots on fruit.
CAUSE: Xanthomonas bacteria. Spread by rain and wind.
TREATMENT: Copper hydroxide or Oxytetracycline bactericide. Apply from petal fall. Avoid overhead irrigation.
URGENCY: HIGH — significant fruit quality reduction.""",
        "ko": """식물: 복숭아 (Prunus persica)
진단: 세균성 구멍병 (Peach Bacterial Spot)
원인: Xanthomonas arboricola pv. pruni (세균)
증상: 잎에 수침상 소반점 → 자갈색화, 황색 테두리, 반점 탈락 후 구멍(탄환 구멍 증상), 과실 함몰 흑색 반점
처방: 수산화동 또는 옥시테트라사이클린 살균제 (꽃잎 낙화 후부터). 두둑 관수 전환.
긴급도: 🔴 높음 — 과실 품질 크게 저하""",
        "es": """PLANTA: Durazno (Prunus persica)
ENFERMEDAD: Mancha bacteriana (Xanthomonas arboricola pv. pruni)
SÍNTOMAS: Manchas acuosas en hojas, halo amarillo, "agujero de bala", manchas hundidas en frutos.
TRATAMIENTO: Hidróxido de cobre u Oxitetraciclina desde caída de pétalos.
URGENCIA: ALTA.""",
    },
    "Healthy Peach": {
        "en": """PLANT: Peach (Prunus persica)
STATUS: Healthy
OBSERVATION: Leaves are uniformly green, no spots or holes, normal fruit development.
RECOMMENDATION: Maintain copper spray program preventatively. Monitor for bacterial spot during wet spring weather.
URGENCY: None — plant is healthy.""",
        "ko": """식물: 복숭아
상태: 정상 (건강)
관찰: 균일한 녹색, 반점·구멍 없음, 과실 정상 발육
권고: 예방 동 살포 프로그램 유지. 습한 봄철 세균성 구멍병 모니터링.
긴급도: 없음 — 정상""",
        "es": """PLANTA: Durazno
ESTADO: Saludable
OBSERVACIÓN: Hojas verde uniforme, sin manchas ni agujeros.
RECOMENDACIÓN: Programa preventivo de cobre. Monitorear en primavera húmeda.
URGENCIA: Ninguna.""",
    },
    # ── Strawberry ──────────────────────────────────────────────────────────
    "Strawberry Leaf Scorch": {
        "en": """PLANT: Strawberry (Fragaria × ananassa)
DISEASE: Leaf Scorch (Diplocarpon earlianum)
SYMPTOMS: Small, irregular dark purple spots on upper leaf surface. Centers may turn gray or tan. Severe infection causes leaves to look "scorched" (burned).
CAUSE: Diplocarpon earlianum fungus. Favored by cool wet weather.
TREATMENT: Captan or Myclobutanil fungicide. Remove old plant debris. Improve air circulation. Use resistant varieties.
URGENCY: MODERATE — reduces plant vigor and yield.""",
        "ko": """식물: 딸기 (Fragaria × ananassa)
진단: 잎소각병 (Strawberry Leaf Scorch)
원인: Diplocarpon earlianum 진균
증상: 잎 윗면 불규칙 암자색 소반점, 중앙 회색~황갈색화, 심하면 잎 전체 소각 외관
처방: 캡탄 또는 마이클로부타닐 살균제. 고엽 제거. 환기 개선. 저항성 품종 재배.
긴급도: ⚠️ 보통 — 생육 및 수량 감소""",
        "es": """PLANTA: Fresa (Fragaria × ananassa)
ENFERMEDAD: Quemadura foliar (Diplocarpon earlianum)
SÍNTOMAS: Manchas irregulares púrpura oscuro, aspecto "quemado" en infección severa.
TRATAMIENTO: Captan o Miclobutanil. Eliminar residuos. Mejorar circulación.
URGENCIA: MODERADA.""",
    },
    "Healthy Strawberry": {
        "en": """PLANT: Strawberry (Fragaria × ananassa)
STATUS: Healthy
OBSERVATION: Leaves are uniformly green with no spots or scorched areas. Normal runner and fruit development.
RECOMMENDATION: Maintain preventative fungicide program. Renew planting every 2-3 years.
URGENCY: None — plant is healthy.""",
        "ko": """식물: 딸기
상태: 정상 (건강)
관찰: 균일한 녹색, 반점·소각 없음, 런너·과실 정상 발육
권고: 예방 살균제 프로그램 유지. 2~3년마다 식재 갱신.
긴급도: 없음 — 정상""",
        "es": """PLANTA: Fresa
ESTADO: Saludable
OBSERVACIÓN: Hojas verde uniforme, sin manchas ni áreas quemadas.
RECOMENDACIÓN: Programa preventivo de fungicidas. Renovar plantación cada 2-3 años.
URGENCIA: Ninguna.""",
    },
    # ── Cherry ──────────────────────────────────────────────────────────────
    "Cherry Powdery Mildew": {
        "en": """PLANT: Cherry (Prunus avium / Prunus cerasus)
DISEASE: Powdery Mildew (Podosphaera clandestina)
SYMPTOMS: White powdery coating on young leaves, shoots and fruit. Infected tissue may curl, distort or turn brown. Reduces fruit quality.
CAUSE: Podosphaera clandestina fungus. Thrives in warm dry conditions with high humidity at night.
TREATMENT: Sulfur or Myclobutanil fungicide. Improve air circulation. Avoid excess nitrogen. Apply at first sign.
URGENCY: MODERATE — can affect fruit quality.""",
        "ko": """식물: 체리 (Prunus avium / Prunus cerasus)
진단: 흰가루병 (Cherry Powdery Mildew)
원인: Podosphaera clandestina 진균
증상: 어린 잎·새순·과실에 흰색 분말 피복, 감염 조직 말리거나 갈변, 과실 품질 저하
처방: 유황 또는 마이클로부타닐 살균제. 환기 개선. 과도한 질소 시비 자제. 첫 징후 시 살포.
긴급도: ⚠️ 보통 — 과실 품질 영향""",
        "es": """PLANTA: Cereza (Prunus avium)
ENFERMEDAD: Oídio (Podosphaera clandestina)
SÍNTOMAS: Recubrimiento polvoriento blanco en hojas jóvenes, brotes y frutos.
TRATAMIENTO: Azufre o Miclobutanil. Mejorar ventilación. Evitar exceso de nitrógeno.
URGENCIA: MODERADA.""",
    },
    "Healthy Cherry": {
        "en": """PLANT: Cherry (Prunus avium / Prunus cerasus)
STATUS: Healthy
OBSERVATION: Leaves are uniformly green, no powdery coating or spots. Normal fruit development.
RECOMMENDATION: Preventative sulfur spray program. Maintain good pruning for air circulation.
URGENCY: None — plant is healthy.""",
        "ko": """식물: 체리
상태: 정상 (건강)
관찰: 균일한 녹색, 분말 피복·반점 없음, 과실 정상 발육
권고: 예방 유황 살포 프로그램. 환기 개선 전정 유지.
긴급도: 없음 — 정상""",
        "es": """PLANTA: Cereza
ESTADO: Saludable
OBSERVACIÓN: Hojas verde uniforme, sin recubrimiento polvoriento ni manchas.
RECOMENDACIÓN: Programa preventivo de azufre. Poda para buena ventilación.
URGENCIA: Ninguna.""",
    },
    # ── Squash ──────────────────────────────────────────────────────────────
    "Squash Powdery Mildew": {
        "en": """PLANT: Squash / Zucchini (Cucurbita pepo)
DISEASE: Powdery Mildew (Podosphaera xanthii / Erysiphe cichoracearum)
SYMPTOMS: White powdery patches on both leaf surfaces. Starts on older leaves, spreads rapidly. Infected leaves turn yellow and die. Reduces yield and quality.
CAUSE: Obligate fungal parasite. Thrives in warm dry weather with high relative humidity.
TREATMENT: Potassium bicarbonate, Neem oil, or Myclobutanil fungicide. Improve air circulation. Avoid overhead watering. Use resistant varieties.
URGENCY: HIGH in warm dry weather — spreads rapidly.""",
        "ko": """식물: 호박 / 애호박 (Cucurbita pepo)
진단: 흰가루병 (Squash Powdery Mildew)
원인: Podosphaera xanthii / Erysiphe cichoracearum (절대기생 진균)
증상: 잎 양면 흰색 분말 반점, 묵은 잎부터 시작 후 빠르게 확산, 감염 잎 황화 후 고사, 수량·품질 감소
처방: 탄산칼륨, 님 오일, 또는 마이클로부타닐 살균제. 환기 개선. 두둑 관수 전환. 저항성 품종 재배.
긴급도: 🔴 고온 건조 시 높음 — 빠른 확산""",
        "es": """PLANTA: Calabacín/Calabaza (Cucurbita pepo)
ENFERMEDAD: Oídio (Podosphaera xanthii)
SÍNTOMAS: Manchas blancas polvorientas en ambas superficies foliares, hojas amarillas y muertas.
TRATAMIENTO: Bicarbonato potásico, aceite de neem o Miclobutanil. Mejorar ventilación.
URGENCIA: ALTA en clima cálido y seco.""",
    },
    # ── Others ──────────────────────────────────────────────────────────────
    "Healthy Blueberry": {
        "en": """PLANT: Blueberry (Vaccinium corymbosum)
STATUS: Healthy
OBSERVATION: Leaves are uniformly green, no spots or mummy berries. Normal growth and fruit set.
RECOMMENDATION: Maintain acidic soil pH (4.5-5.5). Regular monitoring for mummy berry and anthracnose.
URGENCY: None — plant is healthy.""",
        "ko": """식물: 블루베리 (Vaccinium corymbosum)
상태: 정상 (건강)
관찰: 균일한 녹색, 반점·미라 과실 없음, 정상 생육 및 착과
권고: 산성 토양 pH (4.5~5.5) 유지. 미라병 및 탄저병 정기 모니터링.
긴급도: 없음 — 정상""",
        "es": """PLANTA: Arándano
ESTADO: Saludable
OBSERVACIÓN: Hojas verde uniforme, sin manchas.
RECOMENDACIÓN: Mantener pH ácido del suelo (4.5-5.5).
URGENCIA: Ninguna.""",
    },
    "Healthy Raspberry": {
        "en": """PLANT: Raspberry (Rubus idaeus)
STATUS: Healthy
OBSERVATION: Leaves are uniformly green, no spots or cane damage. Normal cane growth.
RECOMMENDATION: Monitor for cane blight and spur blight. Prune out old canes annually.
URGENCY: None — plant is healthy.""",
        "ko": """식물: 라즈베리 (Rubus idaeus)
상태: 정상 (건강)
관찰: 균일한 녹색, 반점·줄기 손상 없음, 정상 줄기 생육
권고: 줄기마름병 및 거름마름병 모니터링. 매년 묵은 줄기 전정.
긴급도: 없음 — 정상""",
        "es": """PLANTA: Frambuesa
ESTADO: Saludable
OBSERVACIÓN: Hojas verde uniforme, sin manchas ni daño en tallos.
URGENCIA: Ninguna.""",
    },
    "Healthy Soybean": {
        "en": """PLANT: Soybean (Glycine max)
STATUS: Healthy
OBSERVATION: Leaves are uniformly trifoliate green, no lesions, no chlorosis. Normal pod development.
RECOMMENDATION: Monitor for soybean rust and sudden death syndrome. Maintain weed control.
URGENCY: None — plant is healthy.""",
        "ko": """식물: 대두 (Glycine max)
상태: 정상 (건강)
관찰: 균일한 3출 녹색 잎, 병반·황화 없음, 꼬투리 정상 발육
권고: 대두 녹병 및 갑작스런 고사 증후군 모니터링. 잡초 방제 유지.
긴급도: 없음 — 정상""",
        "es": """PLANTA: Soja (Glycine max)
ESTADO: Saludable
OBSERVACIÓN: Hojas trifoliadas verde uniforme, sin lesiones.
URGENCIA: Ninguna.""",
    },
    "Orange Citrus Greening": {
        "en": """PLANT: Orange / Citrus (Citrus sinensis)
DISEASE: Citrus Greening (Huanglongbing / HLB — Candidatus Liberibacter asiaticus)
SYMPTOMS: Yellow shoots ("blotchy mottle") asymmetrically distributed on leaf. Lopsided small bitter fruit. Small aborted seeds. Severe decline of entire tree.
CAUSE: Candidatus Liberibacter asiaticus bacterial pathogen spread by Asian citrus psyllid insect.
TREATMENT: No cure. Control psyllid vector with insecticides. Remove and destroy infected trees to prevent spread. Plant certified disease-free nursery trees.
URGENCY: CRITICAL — no cure; remove infected trees to protect healthy ones.""",
        "ko": """식물: 오렌지 / 감귤 (Citrus sinensis)
진단: 황룡병 (Citrus Greening / HLB)
원인: Candidatus Liberibacter asiaticus (세균), 아시아 감귤 깍지벌레가 매개
증상: 잎에 비대칭 황화 (반점 얼룩), 기형·쓴맛 소형 과실, 종자 퇴화, 수목 전체 급격히 쇠퇴
처방: 치료법 없음. 매개충(깍지벌레) 살충제로 방제. 감염목 즉시 제거·소각. 무병 묘목 식재.
긴급도: 🚨 즉시! 치료 불가 — 건강목 보호 위해 감염목 즉시 제거""",
        "es": """PLANTA: Naranja/Cítricos
ENFERMEDAD: HLB/Huanglongbing (Candidatus Liberibacter asiaticus)
SÍNTOMAS: Amarillamiento asimétrico, frutos pequeños amargos, deterioro grave del árbol.
TRATAMIENTO: Sin cura. Controlar vector (psílido). Eliminar árboles infectados.
URGENCIA: CRÍTICA — sin cura disponible.""",
    },
    # ── Additional Tomato ──────────────────────────────────────────────────
    "Tomato Spider Mites": {
        "en": """PLANT: Tomato (Solanum lycopersicum)
DISEASE: Spider Mite damage (Tetranychus urticae — Two-spotted spider mite)
SYMPTOMS: Fine stippling (tiny yellow dots) on upper leaf surface. Bronze/silvery discoloration of leaves. Fine webbing visible on undersides. Severe infestation causes leaf drop.
CAUSE: Two-spotted spider mite (Tetranychus urticae). Pest not a disease. Thrives in hot, dry conditions.
TREATMENT: Miticide (Abamectin, Bifenazate). Spray undersides of leaves. Introduce predatory mites. Increase humidity. Remove heavily infested leaves.
URGENCY: HIGH in hot dry weather — populations can explode rapidly.""",
        "ko": """식물: 토마토
진단: 응애 피해 (Tomato Spider Mites — 점박이응애)
원인: 점박이응애 (Tetranychus urticae), 해충(병이 아님), 고온 건조 조건에서 급증
증상: 잎 윗면 미세 황색 점각, 잎 청동색·은백색 변색, 잎 뒷면 미세 거미줄, 심하면 낙엽
처방: 살응애제(아바멕틴, 비페나제이트). 잎 뒷면에 집중 살포. 포식성 응애 방사. 습도 증가. 심한 잎 제거.
긴급도: 🔴 고온 건조 시 높음 — 개체수 급폭발 가능""",
        "es": """PLANTA: Tomate
ENFERMEDAD: Daño por ácaros (Tetranychus urticae)
SÍNTOMAS: Punteado fino amarillo, decoloración plateada/bronce, telaraña fina en envés.
TRATAMIENTO: Acaricida (Abamectina). Pulverizar envés de hojas. Introducir ácaros depredadores.
URGENCIA: ALTA en clima cálido y seco.""",
    },
    "Tomato Target Spot": {
        "en": """PLANT: Tomato (Solanum lycopersicum)
DISEASE: Target Spot (Corynespora cassiicola)
SYMPTOMS: Circular brown spots with concentric rings and yellow halo on leaves. Spots on fruit appear sunken and brown. Defoliation in severe cases.
CAUSE: Corynespora cassiicola fungus. Favored by warm wet conditions.
TREATMENT: Azoxystrobin or Chlorothalonil fungicide. Remove infected plant material. Improve air circulation.
URGENCY: HIGH — causes defoliation and fruit loss.""",
        "ko": """식물: 토마토
진단: 표적 반점병 (Tomato Target Spot)
원인: Corynespora cassiicola 진균
증상: 잎에 황색 테두리 동심원 갈색 반점, 과실에 함몰 갈색 반점, 심하면 낙엽
처방: 아족시스트로빈 또는 클로로탈로닐 살균제. 감염 식물체 제거. 환기 개선.
긴급도: 🔴 높음 — 낙엽 및 과실 손실""",
        "es": """PLANTA: Tomate
ENFERMEDAD: Mancha diana (Corynespora cassiicola)
SÍNTOMAS: Manchas marrones concéntricas con halo amarillo, manchas hundidas en frutos.
TRATAMIENTO: Azoxistrobina o Clorotalonil. Eliminar material infectado.
URGENCIA: ALTA.""",
    },
    "Tomato Yellow Leaf Curl": {
        "en": """PLANT: Tomato (Solanum lycopersicum)
DISEASE: Tomato Yellow Leaf Curl Virus (TYLCV — Begomovirus)
SYMPTOMS: Upward curling and yellowing of young leaves. Stunted plant growth. Small cupped leaves with yellow edges. Reduced fruit set.
CAUSE: Tomato yellow leaf curl virus (TYLCV) transmitted by whiteflies (Bemisia tabaci).
TREATMENT: No chemical cure for virus. Control whitefly vector with imidacloprid or reflective mulch. Remove and destroy infected plants. Plant TYLCV-resistant varieties.
URGENCY: HIGH — no cure; prevention through vector control critical.""",
        "ko": """식물: 토마토
진단: 황화잎말림바이러스 (TYLCV)
원인: 토마토황화잎말림바이러스(TYLCV), 담배가루이(Bemisia tabaci)가 매개
증상: 어린 잎 상향 말림 및 황화, 식물 왜화, 소형 컵형 잎 황색 테두리, 착과 감소
처방: 바이러스 화학 치료 불가. 이미다클로프리드 또는 반사 멀칭으로 가루이 방제. 감염 식물 즉시 제거. TYLCV 저항성 품종 재배.
긴급도: 🔴 높음 — 치료 불가, 매개충 방제로 예방이 핵심""",
        "es": """PLANTA: Tomate
ENFERMEDAD: Virus del enrollamiento amarillo de la hoja (TYLCV)
SÍNTOMAS: Enrollamiento hacia arriba de hojas jóvenes, amarillamiento, enanismo.
TRATAMIENTO: Sin cura química. Control de mosca blanca con imidacloprid. Eliminar plantas infectadas.
URGENCIA: ALTA.""",
    },
    "Tomato Mosaic Virus": {
        "en": """PLANT: Tomato (Solanum lycopersicum)
DISEASE: Tomato Mosaic Virus (ToMV — Tobamovirus)
SYMPTOMS: Mosaic pattern of light and dark green areas on leaves. Leaf distortion and blistering. Stunted growth. Fruit may show yellow spots or internal browning.
CAUSE: Tomato mosaic virus (ToMV). Highly stable, transmitted mechanically (tools, hands) and through infected seed.
TREATMENT: No chemical cure. Remove infected plants immediately. Sterilize tools with 10% bleach. Plant disease-resistant varieties. Use certified virus-free seed.
URGENCY: HIGH — highly contagious via mechanical contact.""",
        "ko": """식물: 토마토
진단: 토마토 모자이크 바이러스 (ToMV)
원인: 토마토모자이크바이러스(ToMV), 기계적 접촉(도구·손) 및 감염 종자로 전파, 매우 안정적
증상: 잎에 밝은~어두운 녹색 모자이크 무늬, 잎 변형·기포 형성, 왜화, 과실 황색 반점 또는 내부 갈변
처방: 바이러스 화학 치료 불가. 감염 식물 즉시 제거. 10% 염소 소독으로 도구 멸균. 저항성 품종 및 무병 종자 사용.
긴급도: 🔴 높음 — 기계적 접촉으로 고도 전염""",
        "es": """PLANTA: Tomate
ENFERMEDAD: Virus del mosaico del tomate (ToMV)
SÍNTOMAS: Patrón mosaico verde claro-oscuro, distorsión foliar, enanismo.
TRATAMIENTO: Sin cura química. Eliminar plantas infectadas. Esterilizar herramientas con cloro al 10%.
URGENCIA: ALTA.""",
    },
}

# ── instruction 형식 ──────────────────────────────────────────────────────────
INSTRUCTIONS = [
    "A farmer shows you a plant image and asks about this condition: {disease}",
    "What disease does this plant have? The diagnosis is: {disease}",
    "Diagnose this plant and provide treatment guidance. Disease: {disease}",
    "Is this plant healthy or diseased? Label: {disease}",
    "Provide a detailed diagnosis report for this crop condition: {disease}",
    "What should a farmer do if their crop shows: {disease}?",
    "Explain the symptoms and treatment for: {disease}",
]

# ── input 형식 ────────────────────────────────────────────────────────────────
INPUTS_BY_LANG = {
    "en": [
        "Please provide diagnosis and treatment information in English.",
        "Respond in English with full disease details.",
        "Give an English diagnosis with urgency level.",
    ],
    "ko": [
        "Please provide diagnosis and treatment information in Korean.",
        "한국어로 진단 및 처방 정보를 제공해 주세요.",
        "한국 농업인을 위해 한국어로 설명해 주세요.",
    ],
    "es": [
        "Por favor proporcione información de diagnóstico en español.",
        "Responda en español con detalles completos de la enfermedad.",
    ],
}

# ── 데이터셋 생성 ──────────────────────────────────────────────────────────────
dataset = []
random.seed(42)

for disease, lang_data in DISEASE_DB.items():
    for lang, response in lang_data.items():
        inputs = INPUTS_BY_LANG.get(lang, INPUTS_BY_LANG["en"])
        for inst_tmpl in INSTRUCTIONS:
            for inp in inputs:
                instruction = inst_tmpl.format(disease=disease)
                dataset.append({
                    "instruction": instruction,
                    "input": inp,
                    "output": response,
                    "lang": lang,
                    "disease": disease,
                })

# 기존 40개 데이터도 포함
existing = json.load(open(PROJECT_ROOT / "data/gemma4_finetune_dataset.json"))
dataset.extend(existing)

# 중복 제거 (instruction+input 기준)
seen = set()
deduped = []
for item in dataset:
    key = (item["instruction"], item["input"])
    if key not in seen:
        seen.add(key)
        deduped.append(item)

random.shuffle(deduped)

print(f"✅ 총 데이터셋: {len(deduped)}개")
print(f"   질병 종류: {len(DISEASE_DB)}개")
print(f"   언어: {set(item.get('lang','?') for item in deduped)}")

# 저장
out_path = PROJECT_ROOT / "data/gemma4_finetune_v2.json"
json.dump(deduped, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f"   저장: {out_path}")

# 질병별 카운트 확인
from collections import Counter
counts = Counter(item.get('disease','?') for item in deduped)
for d, c in sorted(counts.items(), key=lambda x: -x[1])[:10]:
    print(f"   {d}: {c}개")
