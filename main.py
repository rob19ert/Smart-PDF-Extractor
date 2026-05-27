import pdfplumber
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from unstructured.partition.pdf import partition_pdf
import joblib
import re
from collections import defaultdict
import json
import PIL.Image
import os
import time
import sys
import difflib

import httpx

from dotenv import load_dotenv

def setup_apis():
    """Инициализация API ключей из файла .env"""
    config = {}
    try:
        # Явная загрузка .env файла
        load_dotenv()
            
        or_key = os.getenv("OPENROUTER_API_KEY")
        if or_key:
            config['openrouter_key'] = or_key
            print("✅ OpenRouter API ключ найден!")
        else:
            print("⚠️ OpenRouter API ключ не найден в .env файле.")
            
        return config
    except Exception as e:
        print(f"⚠️ Ошибка при настройке API: {e}")
        return config

def call_openrouter_visual_inspector(image_path, openrouter_key):
    """Вызов OpenRouter (модель poolside/laguna-m.1:free) для проверки разметки."""
    print(f"👁️ Запуск ИИ-инспектора OpenRouter для: {Path(image_path).name}...")
    
    # Модель poolside/laguna-m.1:free на данный момент может не поддерживать vision напрямую через API в бесплатном режиме или иметь специфичный формат.
    # Если модель текстовая, мы передадим описание того, что видим (на основе текущего df), 
    # НО пользователь просил именно для "генерации исправленных изображений" и "поправления ошибок".
    # Для бесплатной модели без Vision мы будем передавать текстовое описание извлеченных блоков.
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": "Smart PDF Extractor Inspector"
    }
    
    prompt = (
        "Ты — эксперт-лингвист и технический инспектор. Я пришлю тебе список извлеченных блоков текста из PDF. "
        "Твоя задача — найти ошибки классификации. "
        "Типы: NarrativeText (обычный текст), Formula (math), Formula (physics), Formula (chemistry).\n"
        "Ошибки для поиска:\n"
        "1. Формула (содержит знаки =, +, -, /, греческие буквы) помечена как NarrativeText.\n"
        "2. Обычный текст (просто слова) помечен как Formula.\n"
        "3. Перепутаны типы формул (например, физическая формула помечена как math).\n\n"
        "Верни ответ ТОЛЬКО в формате строгого JSON списка:\n"
        "[{\"element_id\": \"точная цитата начала блока\", \"correct_type\": \"правильный тип\", \"reason\": \"почему\"}]\n"
        "Если ошибок нет, верни []."
    )

    # Здесь мы будем передавать не картинку (так как laguna-m.1 текстовая), а данные из DF
    # Но так как метод вызывается внутри цикла по страницам, нам нужно получить данные текущей страницы.
    # Для этого передадим данные блоков в контексте.
    return [] # Заглушка, реальный вызов будет реализован в методе класса


# === НАСТРОЙКИ ПО ГОСТУ ===
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


class SmartPDFExtractor:
    # ИИ-модель передается один раз при создании объекта
    def __init__(self, pdf_path, model=None):
        self.pdf_path = Path(pdf_path)
        self.elements = []
        self.model = model

    def extract(self):
        print(f"🚀 Запускаю unstructured для файла: {self.pdf_path.name}...")
        self.elements = partition_pdf(
            filename=str(self.pdf_path),
            strategy="hi_res",
            infer_table_structure=True,
            include_page_breaks=True,
            languages=["rus"],
        )
        print(f"📄 До графового слияния найдено элементов: {len(self.elements)}")
        self.apply_graph_clustering()
        return self.elements

    def apply_graph_clustering(self):
        """
        Графовый метод: выстраивает взаимосвязь между пространственно близкими
        блоками и объединяет разрозненные слова в единые абзацы.
        Это радикально усиливает фактор контекста для классификатора.
        """
        print(" Запуск графового алгоритма связывания смежных блоков...")

        # УДАЛЯЕМ МУСОР: отсекаем блоки, где нет реального текста
        self.elements = [el for el in self.elements if str(el.text).strip()]

        pages = defaultdict(list)
        for el in self.elements:
            page = getattr(el.metadata, 'page_number', 1)
            pages[page].append(el)

        final_elements = []

        for page_num, page_els in pages.items():
            nodes = []
            for el in page_els:
                coords = getattr(el.metadata, 'coordinates', None)
                if coords and coords.points:
                    pts = coords.points
                    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
                    x1, y1 = max(p[0] for p in pts), max(p[1] for p in pts)
                    nodes.append({'el': el, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'height': y1 - y0 })
                else:
                    nodes.append({'el': el, 'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0})

            # Сортировка узлов сверху-вниз, затем слева-направо
            nodes.sort(key=lambda n: (round(n['y0'] / 10), n['x0']))

            n_count = len(nodes)
            adj = {i: [] for i in range(n_count)}

            for i in range(n_count):
                # Ищем соседей в локальном окне (до 20 следующих элементов)
                for j in range(i + 1, min(i + 20, n_count)):
                    n1, n2 = nodes[i], nodes[j]

                    text1 = str(n1['el'].text).strip()
                    text2 = str(n2['el'].text).strip()

                    if text1.endswith(('.', '!', '?', ';', ':')):
                        # Проверяем, что следующий начинается с заглавной
                        if text2 and text2[0].isupper():
                            continue  # Не склеиваем

                    # 2. Короткая строка (не полная строка текста)
                    avg_line_length = 80  # символов
                    if len(text1) < avg_line_length * 0.4:
                        continue  # Возможно конец абзаца

                    # 3. Нумерованный список
                    if re.match(r'^\d+[\.\)]\s', text2):
                        continue

                    vertical_gap = n2['y0'] - n1['y1']
                    font_size = (n1['height'] + n2['height']) / 2
                    
                    # Специальная логика для склеивания многоэтажных формул (дроби, системы уравнений)
                    is_n1_form = 'Formula' in n1['el'].category or self.is_likely_formula(str(n1['el'].text))
                    is_n2_form = 'Formula' in n2['el'].category or self.is_likely_formula(str(n2['el'].text))
                    is_fraction_bar = lambda t: re.fullmatch(r'[-—_]+', str(t).strip()) is not None

                    if (is_n1_form or is_fraction_bar(n1['el'].text)) and (is_n2_form or is_fraction_bar(n2['el'].text)):
                        # Увеличили допустимый разрыв для дробей по вертикали (до 3.0)
                        if -font_size * 3.0 < vertical_gap < font_size * 3.0:
                            x_overlap = min(n1['x1'], n2['x1']) - max(n1['x0'], n2['x0'])
                            if x_overlap > -font_size * 2: # Есть перекрытие или они очень близко по X
                                adj[i].append(j)
                                adj[j].append(i)
                                continue

                    if vertical_gap > font_size * 1.2:  # Отступ больше высоты строки
                        continue

                    left_indent_diff = abs(n1['x0'] - n2['x0'])
                    if left_indent_diff > font_size * 4.0:
                        continue

                    text1, text2 = str(n1['el'].text).strip(), str(n2['el'].text).strip()
                    if re.match(r'^\d{2,4}$', text2) or re.match(r'^P\s*\d+', text2):
                        continue

                    if n1['el'].category == 'Title' or n2['el'].category == 'Title':
                        if vertical_gap > font_size * 0.8:
                            continue

                    # Пропускаем элементы без валидных координат
                    if n1['x1'] == 0 or n2['x1'] == 0: continue

                    # Связываем только текстовые элементы
                    txt_cats = ['NarrativeText', 'UncategorizedText', 'Title']
                    if n1['el'].category not in txt_cats or n2['el'].category not in txt_cats:
                        continue

                    y_overlap = min(n1['y1'], n2['y1']) - max(n1['y0'], n2['y0'])
                    is_same_line = (y_overlap > -font_size * 0.3) and (n2['x0'] > n1['x0']) and (
                                n2['x0'] - n1['x1'] < font_size * 2.5)

                    is_next_line = (0 <= vertical_gap < font_size * 0.6) and (
                                abs(n1['x0'] - n2['x0']) < font_size * 4.0)

                    # Защита от слипания разных абзацев: если следующая строка имеет отступ вправо (красная строка),
                    # это начало НОВОГО абзаца. Мы их не объединяем.
                    if is_next_line and not is_same_line:
                        if n2['x0'] - n1['x0'] > font_size * 1.2:
                            is_next_line = False

                    # Доп. проверка: не связываем разные типы контента (если только это не инлайн-формула)
                    type1 = n1['el'].category
                    type2 = n2['el'].category
                    
                    # НОВОЕ: Проактивная защита формул, даже если unstructured назвал их текстом
                    is_f1 = 'Formula' in type1 or self.is_likely_formula(text1)
                    is_f2 = 'Formula' in type2 or self.is_likely_formula(text2)
                    
                    # Если один блок - формула, а второй - текст, разрываем связь по вертикали
                    if (is_f1 and not is_f2) or (not is_f1 and is_f2):
                        if vertical_gap > font_size * 0.2: # Даже минимальный отступ означает, что это разные блоки
                            continue
                            
                    if ('Formula' in type1 and 'Text' in type2) or ('Text' in type1 and 'Formula' in type2):
                        if vertical_gap > font_size * 0.3:
                            continue

                    if (is_same_line or is_next_line):
                        adj[i].append(j)
                        adj[j].append(i)

            # Поиск компонент связности (DFS)
            visited = set()
            components = []

            for i in range(n_count):
                if i not in visited:
                    comp = []
                    stack = [i]
                    while stack:
                        curr = stack.pop()
                        if curr not in visited:
                            visited.add(curr)
                            comp.append(curr)
                            stack.extend(adj[curr])
                    
                    # Восстанавливаем порядок чтения внутри объединенной компоненты
                    comp.sort(key=lambda idx: (round(nodes[idx]['y0'] / 10), nodes[idx]['x0']))
                    components.append(comp)

            # Объединение элементов внутри каждой компоненты
            for comp in components:
                if len(comp) == 1:
                    final_elements.append(nodes[comp[0]]['el'])
                else:
                    base_el = nodes[comp[0]]['el']
                    merged_text = " ".join([str(nodes[idx]['el'].text).strip() for idx in comp])
                    merged_text = re.sub(r'\s+', ' ', merged_text) # Очистка от лишних пробелов
                    
                    base_el.text = merged_text
                    base_el.category = 'NarrativeText'

                    # Пересчет общей рамки (Bounding Box) для визуализации
                    new_x0 = min(nodes[idx]['x0'] for idx in comp)
                    new_y0 = min(nodes[idx]['y0'] for idx in comp)
                    new_x1 = max(nodes[idx]['x1'] for idx in comp)
                    new_y1 = max(nodes[idx]['y1'] for idx in comp)

                    if hasattr(base_el.metadata, 'coordinates') and base_el.metadata.coordinates:
                        sys = base_el.metadata.coordinates.system
                        CoordClass = type(base_el.metadata.coordinates)
                        base_el.metadata.coordinates = CoordClass(
                            points=((new_x0, new_y0), (new_x1, new_y0), (new_x1, new_y1), (new_x0, new_y1)),
                            system=sys
                        )
                    final_elements.append(base_el)

        # Итоговая сортировка: по страницам, затем сверху вниз (по Y)
        self.elements = sorted(final_elements, key=lambda el: (
            getattr(el.metadata, 'page_number', 0) or 0,
            getattr(el.metadata, 'coordinates').points[0][1] if getattr(el.metadata, 'coordinates', None) and getattr(el.metadata, 'coordinates').points else 0
        ))
        print(f"✅ После графовой кластеризации объединено элементов: {len(self.elements)}")

    def is_likely_formula(self, text):
                """Универсальный структурный анализатор. Ищет высокую концентрацию цифр и спецсимволов."""
                if not text or len(text) > 400:
                    return False

                text = text.strip()
                clean_text = text.replace(" ", "")
                if not clean_text: return False

                if re.match(r'^\d{2,4}$', text):
                    return False
                if re.match(r'^[PpСс]\.?\s*\d+$', text):
                    return False
                if re.match(r'^[-\d]+\s*[PpСс]\.?\s*\d*$', text):
                    return False

                if re.search(r'[А-Я]{2,}\s*[\d.-]+', text):
                    return False

                if re.search(r'[А-Я]{2,}\s*\d+', text):  # ГОСТ 12345, СНиП 2.01.01
                    return False

                russian_words = re.findall(r'[А-Яа-яЁё]{4,}', text)
                if len(russian_words) >= 3:
                    russian_chars = sum(1 for c in text if 'А' <= c <= 'я' or c == 'ё')
                    if russian_chars / len(text) > 0.4:
                        return False

                if text.isupper() and len(text) > 10:
                    return False

                tech_symbols = set('=+-*/^\\_→Δ∫∑°≈·|[]{}()<>!')
                tech_count = sum(1 for c in clean_text if c in tech_symbols)
                digit_count = sum(1 for c in clean_text if c.isdigit())
                alpha_count = sum(1 for c in clean_text if c.isalpha())
                cyrillic_count = sum(1 for c in text if 'А' <= c <= 'я')

                if alpha_count == 0:
                    return (tech_count + digit_count) > 0

                density = (tech_count + digit_count) / alpha_count

                has_equation = any(c in clean_text for c in ['=', '→', '≈', '>', '<'])

                has_latex = any(c in clean_text for c in ['\\', '^', '_', '{', '}'])

                if has_latex and (tech_count + digit_count) >= 2:
                    return True

                if re.fullmatch(r'\d+', clean_text) or re.fullmatch(r'[\(\[]\d+[\)\]]', clean_text):
                    return False

                if text.isupper() and len(text) > 10:
                    return False

                if cyrillic_count / len(text) > 0.5 and len(text) > 20:
                    return False

               # Решение:
                if density > 0.45: return True # Высокая плотность - точно формула
                
                # НОВОЕ: Отлов коротких уравнений (типа x = y + 1)
                if has_equation and (tech_count + digit_count) > 0 and cyrillic_count < 10: 
                    return True 
                    
                if has_equation and density >= 0.2: return True 
                if has_equation and len(text) < 80 and cyrillic_count < 15: return True 
                if has_latex and (tech_count + digit_count) >= 1: return True
                return False

    def is_table_of_contents(self, elements):
        """Проверяет, является ли группа элементов оглавлением"""
        text = " ".join([str(el.text) for el in elements])

        # Признаки оглавления
        has_page_numbers = len(re.findall(r'\s+(\d+)\s*$', text, re.MULTILINE)) > 3
        has_dots = '...' in text or '…' in text
        has_chapter_pattern = re.search(r'(Глава|Chapter|Раздел|§)\s*\d+', text)

        # Не должно быть формул
        formula_count = sum(1 for el in elements if 'Formula' in str(el.category))
        table_count = sum(1 for el in elements if el.category == 'Table')

        # Оглавление, если есть страницы, точки/многоточие и НЕТ формул/таблиц
        if (has_page_numbers or has_dots or has_chapter_pattern) and formula_count == 0:
            return True

        return False

    def is_table_with_formulas(self, element):
        """Проверяет, содержит ли таблица формулы"""
        if element.category != 'Table':
            return False

        text = str(element.text)
        formula_indicators = sum([
            1 for c in text if c in '=+-*/^_→∫∑'
        ])

        if formula_indicators > 5:
            # Это не обычная таблица, а таблица с формулами
            element.category = 'FormulaTable'
            return True
        return False

    def to_dataframe(self):

        page_elements = defaultdict(list)
        for el in self.elements:
            page = getattr(el.metadata, 'page_number', 1)
            page_elements[page].append(el)

        for page, els in page_elements.items():
            # Поиск групп таблиц, которые могут быть оглавлением
            table_groups = []
            current_group = []

            for i, el in enumerate(els):
                if el.category == 'Table':
                    current_group.append(el)
                else:
                    if current_group and self.is_table_of_contents(current_group):
                        for table_el in current_group:
                            table_el.category = 'TableOfContents'
                    current_group = []

            if current_group and self.is_table_of_contents(current_group):
                for table_el in current_group:
                    table_el.category = 'TableOfContents'

        data = []
        for i, el in enumerate(self.elements):
            text = str(el.text).strip()
            if not text:
                continue

            category_name = el.category

            # Эвристика: Крупные графики (по размеру рамки)
            if hasattr(el.metadata, 'coordinates') and el.metadata.coordinates:
                pts = el.metadata.coordinates.points
                h = max(p[1] for p in pts) - min(p[1] for p in pts)
                # Если блок очень высокий (сотни пикселей), но букв мало - это картинка/схема
                if h > 200 and len(text) < 150: 
                    category_name = 'Figure'
                    el.category = category_name
                    el.is_manually_corrected = True

            # Эвристика: Подписи к рисункам и таблицам
            if re.match(r'^(Рис\.|Рисунок|Таблица|Fig\.|Table)\s*\d+', text, re.IGNORECASE):
                category_name = 'Caption'
                el.category = category_name
                el.is_manually_corrected = True

            # === МАГИЯ ИИ: Сбор контекста и классификация формулы ===
            is_formula_candidate = (category_name == 'Formula') or \
                                   (category_name in ['NarrativeText', 'UncategorizedText'] and self.is_likely_formula(text))

            # Пропускаем ML классификацию, если блок был исправлен вручную ИИ-инспектором
            if getattr(el, 'is_manually_corrected', False):
                 category_name = el.category # Оставляем то, что установил инспектор
            elif is_formula_candidate and self.model:
                try:
                    # Смарт-контекст (2 блока ДО и до 5 блоков ПОСЛЕ)
                    before_blocks = [str(self.elements[j].text) for j in range(max(0, i - 2), i)]
                    context_before = " ".join(before_blocks)
                    
                    after_blocks = []
                    for j in range(i + 1, min(len(self.elements), i + 6)):
                        after_text = str(self.elements[j].text).strip()
                        if not after_text: continue
                        after_blocks.append(after_text)
                        if len(after_text) > 250: break
                    context_after = " ".join(after_blocks)
                    context_combined = f"{context_before} {context_after}"

                    # 1. Предсказание ML (базовое)
                    input_df = pd.DataFrame([{'latex_code': text, 'context': context_combined}])
                    predicted_class = self.model.predict(input_df)[0]
                    
                    # 2. СТРУКТУРНЫЙ ВЕСОВОЙ АНАЛИЗ (Tie-breaker)
                    weights = {'physics': 0, 'chemistry': 0, 'math': 0}
                    formula_orig = text
                    formula_lower = text.lower()
                    context_lower = context_combined.lower()
                    
                    # Полный список химических элементов
                    periodic_table = {
                        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                        'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
                        'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
                        'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                        'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
                    }
                    
                    # Символы, которые могут быть и в физике, и в химии
                    ambiguous_elements = {'F', 'H', 'P', 'V', 'S', 'C', 'O', 'K', 'I', 'N', 'W', 'U', 'T'}
                    
                    # Поиск всех потенциальных элементов в формуле
                    found_elements = re.findall(r'\b[A-Z][a-z]?\b', formula_orig)
                    chemistry_match = False
                    
                    for chem_el in found_elements:
                        if chem_el in periodic_table:
                            if chem_el in ambiguous_elements:
                                # Если символ спорный, смотрим на контекст
                                chem_keywords = ['реакц', 'соединени', 'веществ', 'молекул', 'атом', 'раствор', 'сплав', 'легир', 'оксид']
                                phys_keywords = ['сила', 'напряжен', 'давлен', 'объем', 'энерги', 'мощн', 'ток', 'скорост', 'ускорен']
                                
                                chem_score = sum(1 for k in chem_keywords if k in context_lower)
                                phys_score = sum(1 for k in phys_keywords if k in context_lower)
                                
                                if chem_score > phys_score:
                                    weights['chemistry'] += 8
                                    chemistry_match = True
                                elif phys_score > chem_score:
                                    weights['physics'] += 8
                                else:
                                    # Если контекст молчит, доверяем базовому предсказанию или добавляем чуть-чуть веса
                                    weights['chemistry'] += 2
                            else:
                                # Однозначные элементы (Fe, Mg, Al...) сразу дают большой вес химии
                                weights['chemistry'] += 10
                                chemistry_match = True

                    # Дополнительные признаки химии (стрелки, индексы внизу)
                    if any(m in formula_lower for m in ['\\rightarrow', '→', '∆h', '∆g']):
                        weights['chemistry'] += 10
                    
                    # --- ПРИЗНАКИ ФИЗИКИ ---
                    if any(c in formula_lower for c in ['\\sigma', '\\tau', '\\varepsilon', '\\mu', '\\rho', '\\phi', '\\lambda']):
                        weights['physics'] += 8
                        
                    # --- ПРИЗНАКИ МАТЕМАТИКИ ---
                    if any(c in formula_lower for c in ['\\sum', '\\int', '\\partial', '\\infty', '∫', '∑']):
                        weights['math'] += 8

                    # --- КОНТЕКСТНЫЕ ГИРЬКИ (Общие - УСИЛЕННЫЕ) ---
                    phys_keywords = ['деформац', 'пластич', 'напряжен', 'прочност', 'модуль', 'упруг', 'вязкост', 'давлен', 'тепло', 'скорост', 'ускорен']
                    if any(w in context_lower for w in phys_keywords): weights['physics'] += 15

                    chem_keywords = ['реакци', 'оксид', 'сплав', 'легирован', 'раствор', 'молекул', 'атом']
                    if any(w in context_lower for w in chem_keywords): weights['chemistry'] += 15

                    math_keywords = ['статистик', 'вероятност', 'матриц', 'векторн', 'предел', 'производн']
                    if any(w in context_lower for w in math_keywords): weights['math'] += 15

                    # Результирующий класс: 
                    max_weight = max(weights.values())
                    if max_weight >= 5:
                        predicted_class = max(weights, key=weights.get)

                    category_name = f"Formula ({predicted_class})"
                    el.category = category_name
                except Exception as e:
                    print(f"Ошибка классификации: {e}")

            data.append({
                'page': getattr(el.metadata, 'page_number', 0),
                'type': category_name,
                'text_preview': text[:150] + '...' if len(text) > 150 else text,
                'text': text,
                'element_id': i
            })

        df = pd.DataFrame(data)
        excel_filename = f"{self.pdf_path.stem}_blocks_final.xlsx"
        df.to_excel(excel_filename, index=False)
        print(f"💾 Сохранено: {excel_filename}")
        return df

    def visualize(self, page_number, output_image=None):
        page_blocks = [el for el in self.elements
                       if getattr(el.metadata, 'page_number', None) == page_number]

        if not page_blocks: return

        with pdfplumber.open(self.pdf_path) as pdf:
            if page_number > len(pdf.pages): return
            page = pdf.pages[page_number - 1]
            img = page.to_image(resolution=200)
            page_img = np.array(img.original)

        width_px, height_px = img.original.width, img.original.height
        fig, ax = plt.subplots(figsize=(10, 14.14), dpi=200)
        ax.imshow(page_img, origin='upper')
        ax.set_xlim(0, width_px); ax.set_ylim(height_px, 0)
        ax.set_autoscale_on(False)

        type_map = {
            'NarrativeText': 'text', 'UncategorizedText': 'text', 'Title': 'text',
            'Formula': 'formula', 'Formula (physics)': 'phys',
            'Formula (chemistry)': 'chem', 'Formula (math)': 'math',
            'Table': 'table', 'Caption': 'caption'
        }
        colors = {
            'text': '#4285F4', 'formula': '#34A853',
            'phys': '#0F9D58', 'chem': '#00E676', 'math': '#B9F6CA',
            'table': '#EA4335', 'caption': '#FBBC05'
        }


        for el in page_blocks:
            if not hasattr(el.metadata, 'coordinates') or not el.metadata.coordinates: continue
            coords = el.metadata.coordinates.points
            xs, ys = [p[0] for p in coords], [p[1] for p in coords]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            sys = el.metadata.coordinates.system
            u_w, u_h = (sys.width, sys.height) if sys and hasattr(sys, 'width') else (page.width, page.height)
            rx, rw = (x0 / u_w) * width_px, ((x1 - x0) / u_w) * width_px
            if sys and "Bottom" in sys.__class__.__name__:
                ry, rh = (1.0 - (y1 / u_h)) * height_px, ((y1 - y0) / u_h) * height_px
            else:
                ry, rh = (y0 / u_h) * height_px, ((y1 - y0) / u_h) * height_px

            el_type = type_map.get(el.category, 'text')
            color = colors.get(el_type, '#4285F4')
            ax.add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor=color, facecolor=color, alpha=0.15))
            if rw > 40 and rh > 15:
                # Подписи выравниваем по левому краю страницы (x = 10)
                ax.text(10, ry, f" {el_type} ", color='black' if 'math' in el_type else 'white',
                        fontsize=10, weight='bold', bbox=dict(boxstyle='round', facecolor=color, alpha=0.9), ha='left', va='top')

        # Добавляем легенду цветов
        legend_elements = [
            patches.Patch(facecolor='#4285F4', edgecolor='#4285F4', alpha=0.5, label='Текст'),
            patches.Patch(facecolor='#34A853', edgecolor='#34A853', alpha=0.5, label='Формула (Общая)'),
            patches.Patch(facecolor='#0F9D58', edgecolor='#0F9D58', alpha=0.5, label='Формула (phys)'),
            patches.Patch(facecolor='#00E676', edgecolor='#00E676', alpha=0.5, label='Формула (chem)'),
            patches.Patch(facecolor='#B9F6CA', edgecolor='#B9F6CA', alpha=0.5, label='Формула (math)'),
            patches.Patch(facecolor='#EA4335', edgecolor='#EA4335', alpha=0.5, label='Таблица'),
            patches.Patch(facecolor='#FBBC05', edgecolor='#FBBC05', alpha=0.5, label='Подпись')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1), title="Типы блоков", frameon=True)

        ax.axis('off')
        if output_image:
            plt.savefig(output_image, dpi=200, bbox_inches='tight', pad_inches=0.1)
            print(f"🖼️ Изображение успешно обновлено: {Path(output_image).name}")
        plt.close()

    def inspect_with_openrouter(self, page_number, openrouter_key):
        """Инспекция разметки страницы с помощью текстовой модели OpenRouter."""
        if not openrouter_key: return []
        
        print(f"👁️ Запуск ИИ-инспектора OpenRouter для страницы {page_number}...")
        
        page_blocks = [el for el in self.elements 
                       if getattr(el.metadata, 'page_number', 1) == page_number]
        
        if not page_blocks: return []

        # Формируем компактное текстовое представление блоков для анализа
        blocks_data = []
        # Разрешенные категории для проверки ИИ (остальные он не трогает)
        allowed_cats = ['NarrativeText', 'UncategorizedText', 'Formula', 
                        'Formula (math)', 'Formula (physics)', 'Formula (chemistry)']
        
        for el in page_blocks:
            if el.category not in allowed_cats:
                continue # ПРОПУСКАЕМ Figure, Caption, Table, чтобы ИИ их не сломал
                
            text = str(el.text).strip()
            if not text: continue
            blocks_data.append({
                "id": self.elements.index(el),
                "current_type": el.category,
                "text": text[:200] # Даем достаточно контекста
            })

        prompt = (
            "Ты — эксперт-лингвист и технический инспектор. Ниже приведен список блоков, извлеченных из технического PDF-документа. "
            "Твоя задача — проверить правильность их классификации по типам: NarrativeText (текст), Formula (math), Formula (physics), Formula (chemistry).\n"
            "Признаки формул:\n"
            "- math: чистая математика, функции, суммы, интегралы.\n"
            "- physics: формулы с греческими буквами (sigma, tau), физическими константами, переменными (F, v, P, E).\n"
            "- chemistry: химические элементы (Fe, H2O, C, O2), знаки реакций (стрелки, плюс).\n"
            "ОШИБКОЙ считается, если сложная формула помечена как NarrativeText или если обычные слова помечены как Formula.\n\n"
            "Верни ответ ТОЛЬКО в формате строгого JSON списка, без лишнего текста:\n"
            "[{\"id\": 123, \"correct_type\": \"правильный тип\", \"reason\": \"почему\"}]\n"
            "Если всё верно, верни [].\n\n"
            f"ДАННЫЕ СТРАНИЦЫ {page_number}:\n"
            f"{json.dumps(blocks_data, ensure_ascii=False)}"
        )

        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "poolside/laguna-m.1:free",
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = httpx.post(url, headers=headers, json=data, timeout=60.0)
            res_json = response.json()
            
            if 'choices' in res_json:
                content = res_json['choices'][0]['message']['content']
                # Извлекаем JSON из ответа
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].strip()
                
                content = content.replace('\\', '\\\\')
                return json.loads(content)
        except Exception as e:
            print(f"⚠️ Ошибка при обращении к OpenRouter: {e}")
        
        return []

    def apply_corrections(self, corrections, df):
        if not corrections: return df
        print(f"🔧 Применение {len(corrections)} ИИ-исправлений...")
        dataset_file = "retrain_dataset.csv"
        
        for corr in corrections:
            idx = corr.get('id')
            correct_type = corr.get('correct_type', '')
            reason = corr.get('reason', '')
            
            # Точное применение по индексу
            if isinstance(idx, int) and 0 <= idx < len(self.elements):
                el = self.elements[idx]
                if el.category != correct_type:
                    print(f"   -> Исправлен блок[{str(el.text)[:30]}...]: {el.category} -> {correct_type} ({reason})")
                    el.category = correct_type
                    el.is_manually_corrected = True # Флаг защиты от перезаписи
                    
                    data_row = pd.DataFrame([{'text': str(el.text), 'correct_class': correct_type, 'reason': reason}])
                    data_row.to_csv(dataset_file, mode='a' if os.path.exists(dataset_file) else 'w', header=not os.path.exists(dataset_file), index=False)
                    
        return self.to_dataframe()

if __name__ == "__main__":
    api_config = setup_apis()
    or_key = api_config.get('openrouter_key')
    
    try:
        global_model = joblib.load('formula_classifier.pkl')
        print("🧠 ИИ-классификатор формул успешно загружен!")
    except:
        global_model = None
        print("⚠️ Модель не найдена.")

    output_dir = Path("output_images")
    output_dir.mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        pdf_files = [Path(target_file)]
        if not pdf_files[0].exists():
            print(f"❌ Файл {target_file} не найден!")
            sys.exit(1)
    else:
        pdf_files = list(Path(".").glob("*.pdf"))

    for pdf_file in pdf_files:
        print(f"\n📄 Обработка: {pdf_file.name}")
        extractor = SmartPDFExtractor(pdf_file, model=global_model)
        extractor.extract()
        
        df = extractor.to_dataframe()
        pages = {int(getattr(el.metadata, 'page_number', 1)) for el in extractor.elements 
                 if getattr(el.metadata, 'page_number', None)}
        
        has_corrections = False
        all_corrections = []
        
        for p in pages:
            image_path = output_dir / f"{pdf_file.stem}_page{p}_initial.png"
            final_image_path = output_dir / f"{pdf_file.stem}_page{p}.png"
            extractor.visualize(p, str(image_path))
            
            if or_key:
                corrections = extractor.inspect_with_openrouter(p, or_key)
                if corrections:
                    print(f"🔍 Найдено потенциальных исправлений: {len(corrections)}")
                    all_corrections.extend(corrections)
                    has_corrections = True
            
            if not has_corrections and os.path.exists(image_path):
                if os.path.exists(final_image_path): os.remove(final_image_path)
                os.replace(image_path, final_image_path)

        if has_corrections:
            print("🔄 Применение ИИ-исправлений и перерисовка...")
            df = extractor.apply_corrections(all_corrections, df)
            for p in pages:
                temp_img = output_dir / f"{pdf_file.stem}_page{p}_initial.png"
                final_img = output_dir / f"{pdf_file.stem}_page{p}.png"
                extractor.visualize(p, str(final_img))
                if os.path.exists(temp_img): os.remove(temp_img)
