"""
Модуль для сегментации изображений рукописного текста на отдельные строки
Поддерживает несколько методов сегментации:
- YOLOv8: использует модель armvectores/yolov8n_handwritten_text_detection для детекции текста
- PaddleOCR: использует PaddleOCR для детекции текстовых регионов
- EasyOCR: использует EasyOCR для детекции текстовых регионов
- Kraken: использует Kraken с моделью blla для сегментации страницы
- Contours: метод на основе контуров OpenCV
- Projection: горизонтальная проекция с бинаризацией и морфологическими операциями
- Auto: автоматически выбирает доступный метод в порядке приоритета
"""
import os
import cv2
import numpy as np
from PIL import Image


def segment_with_kraken(image_path):
    """
    Сегментация изображения на строки с использованием Kraken и модели blla
    
    Args:
        image_path: путь к изображению
        
    Returns:
        список PIL Image объектов - изображения отдельных строк
    """
    try:
        from kraken import pageseg
        
        # Загружаем изображение
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        # Конвертируем PIL Image в формат, который понимает Kraken
        # Kraken работает с изображениями в формате (height, width)
        if len(img_array.shape) == 3:
            # RGB изображение - конвертируем в grayscale
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        # Сегментация страницы на строки
        # pageseg.segment возвращает словарь с ключами 'lines', 'text_direction', и т.д.
        try:
            # Пробуем использовать встроенную модель blla (baseline detection)
            seg = pageseg.segment(img_gray)
        except Exception as e:
            # Если не получается, пробуем альтернативный метод
            print(f"  Предупреждение: Kraken pageseg.segment не сработал: {e}")
            try:
                # Пробуем без модели
                seg = pageseg.segment(img_gray, model=None)
            except:
                return None
        
        # Извлекаем регионы строк
        line_images = []
        if seg and 'lines' in seg:
            for line_region in seg['lines']:
                # Получаем координаты региона строки
                # Kraken возвращает строки в формате словаря или объекта с атрибутами
                try:
                    if isinstance(line_region, dict):
                        # Если line_region - словарь с координатами
                        x0 = int(line_region.get('x0', 0))
                        y0 = int(line_region.get('y0', 0))
                        x1 = int(line_region.get('x1', img_gray.shape[1]))
                        y1 = int(line_region.get('y1', img_gray.shape[0]))
                    elif hasattr(line_region, 'bbox'):
                        # Если line_region - объект с атрибутом bbox
                        bbox = line_region.bbox
                        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    elif hasattr(line_region, '__getitem__'):
                        # Если line_region поддерживает индексацию
                        x0, y0, x1, y1 = int(line_region[0]), int(line_region[1]), int(line_region[2]), int(line_region[3])
                    elif isinstance(line_region, (list, tuple)) and len(line_region) >= 4:
                        # Если line_region - список/кортеж [x0, y0, x1, y1]
                        x0, y0, x1, y1 = int(line_region[0]), int(line_region[1]), int(line_region[2]), int(line_region[3])
                    else:
                        # Пробуем получить атрибуты напрямую
                        x0 = int(getattr(line_region, 'x0', 0))
                        y0 = int(getattr(line_region, 'y0', 0))
                        x1 = int(getattr(line_region, 'x1', img_gray.shape[1]))
                        y1 = int(getattr(line_region, 'y1', img_gray.shape[0]))
                    
                    # Обрезаем изображение по региону строки
                    # Добавляем небольшой отступ для лучшего качества
                    padding = 5
                    x0 = max(0, x0 - padding)
                    y0 = max(0, y0 - padding)
                    x1 = min(img_gray.shape[1], x1 + padding)
                    y1 = min(img_gray.shape[0], y1 + padding)
                    
                    if x1 > x0 and y1 > y0:
                        line_img = img_gray[y0:y1, x0:x1]
                        # Конвертируем обратно в PIL Image
                        line_pil = Image.fromarray(line_img).convert('RGB')
                        line_images.append(line_pil)
                except Exception as e:
                    # Пропускаем строку, если не удалось извлечь координаты
                    continue
        
        return line_images if line_images else None
        
    except ImportError:
        # Kraken не установлен
        return None
    except Exception as e:
        print(f"  Ошибка при сегментации с Kraken: {e}")
        import traceback
        traceback.print_exc()
        return None


def segment_with_projection(image_path):
    """
    Сегментация изображения на строки с использованием горизонтальной проекции
    (Fallback метод, если Kraken недоступен)
    
    Args:
        image_path: путь к изображению
        
    Returns:
        список PIL Image объектов - изображения отдельных строк
    """
    try:
        # Загружаем изображение
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        # Конвертируем в grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Улучшаем контраст для лучшей бинаризации
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Бинаризация (адаптивный порог для лучшей работы с разным освещением)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Морфологические операции для соединения символов в строки
        # Создаем горизонтальное ядро для соединения символов в строки
        # Размер ядра зависит от размера изображения
        kernel_width = max(30, int(gray.shape[1] * 0.1))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        dilated = cv2.dilate(binary, horizontal_kernel, iterations=1)
        
        # Горизонтальная проекция (сумма пикселей по каждой строке)
        horizontal_projection = np.sum(dilated, axis=1)
        
        # Сглаживаем проекцию для лучшего обнаружения строк
        try:
            from scipy import ndimage
            smoothed_projection = ndimage.gaussian_filter1d(horizontal_projection.astype(float), sigma=2)
        except ImportError:
            # Fallback: используем простое сглаживание через свертку с гауссовым ядром
            # Создаем простое гауссово ядро для сглаживания
            kernel_size = 5
            sigma = 2.0
            kernel = np.exp(-0.5 * ((np.arange(kernel_size) - kernel_size // 2) / sigma) ** 2)
            kernel = kernel / kernel.sum()
            # Применяем свертку
            smoothed_projection = np.convolve(horizontal_projection.astype(float), kernel, mode='same')
        
        # Находим строки используя более умный подход
        # Ищем локальные максимумы и минимумы
        max_val = np.max(smoothed_projection)
        mean_val = np.mean(smoothed_projection)
        std_val = np.std(smoothed_projection)
        
        # Динамический порог на основе статистики
        # Порог должен быть выше среднего, но не слишком высоким
        threshold = max(mean_val + std_val * 0.5, max_val * 0.15)
        
        # Находим регионы с текстом (выше порога)
        text_mask = smoothed_projection > threshold
        
        # Находим границы строк (переходы от текста к пустоте и обратно)
        lines = []
        in_line = False
        line_start = 0
        
        # Минимальная высота строки (зависит от размера изображения)
        min_line_height = max(10, int(gray.shape[0] * 0.015))  # Уменьшено для лучшего обнаружения
        # Минимальный промежуток между строками (более агрессивный)
        min_gap = max(3, int(gray.shape[0] * 0.008))  # Уменьшено для лучшего разделения
        
        # Анализируем распределение значений проекции для определения типичной высоты строки
        # Находим локальные максимумы в проекции
        local_maxima = []
        for i in range(1, len(smoothed_projection) - 1):
            if (smoothed_projection[i] > smoothed_projection[i-1] and 
                smoothed_projection[i] > smoothed_projection[i+1] and
                smoothed_projection[i] > threshold):
                local_maxima.append(i)
        
        # Если нашли максимумы, оцениваем типичную высоту строки
        if len(local_maxima) > 1:
            # Вычисляем среднее расстояние между максимумами
            distances = [local_maxima[i+1] - local_maxima[i] for i in range(len(local_maxima)-1)]
            if distances:
                avg_line_height = np.median(distances)
                min_line_height = max(10, int(avg_line_height * 0.3))  # 30% от средней высоты
                min_gap = max(3, int(avg_line_height * 0.15))  # 15% от средней высоты
        
        for i in range(len(text_mask)):
            if not in_line and text_mask[i]:
                # Начало строки
                line_start = i
                in_line = True
            elif in_line and not text_mask[i]:
                # Проверяем, достаточно ли большой промежуток (конец строки)
                # Считаем количество последовательных пустых строк
                gap_size = 0
                for j in range(i, min(i + min_gap * 2, len(text_mask))):  # Увеличиваем окно поиска
                    if not text_mask[j]:
                        gap_size += 1
                    else:
                        break
                
                if gap_size >= min_gap:
                    # Достаточно большой промежуток - конец строки
                    line_end = i
                    if line_end - line_start >= min_line_height:
                        lines.append((line_start, line_end))
                    in_line = False
                # Если промежуток маленький, продолжаем строку
        
        # Если строка продолжается до конца изображения
        if in_line:
            line_end = len(text_mask) - 1
            if line_end - line_start >= min_line_height:
                lines.append((line_start, line_end))
        
        # Если не нашли строки, пробуем более низкий порог
        if len(lines) == 0:
            threshold = max(mean_val, max_val * 0.1)
            text_mask = smoothed_projection > threshold
            in_line = False
            line_start = 0
            
            for i in range(len(text_mask)):
                if not in_line and text_mask[i]:
                    line_start = i
                    in_line = True
                elif in_line and not text_mask[i]:
                    gap_size = 0
                    for j in range(i, min(i + min_gap, len(text_mask))):
                        if not text_mask[j]:
                            gap_size += 1
                        else:
                            break
                    
                    if gap_size >= min_gap:
                        line_end = i
                        if line_end - line_start >= min_line_height:
                            lines.append((line_start, line_end))
                        in_line = False
            
            if in_line:
                line_end = len(text_mask) - 1
                if line_end - line_start >= min_line_height:
                    lines.append((line_start, line_end))
        
        # Если все еще одна строка или слишком мало строк, пробуем найти локальные минимумы
        # Проверяем, если одна строка занимает больше 20% изображения или если строк меньше ожидаемого
        expected_lines = max(2, int(gray.shape[0] / (min_line_height * 2)))
        single_line_too_large = len(lines) == 1 and lines[0][1] - lines[0][0] > gray.shape[0] * 0.2
        
        # Принудительное разделение, если нашли только одну большую строку
        if single_line_too_large:
            print(f"  Обнаружена одна большая строка ({lines[0][1] - lines[0][0]}px), принудительно разделяем...")
            # Используем более агрессивный метод - делим на равные части с поиском минимумов
            original_line = lines[0]
            line_projection = horizontal_projection[original_line[0]:original_line[1]]
            
            # Оцениваем примерное количество строк на основе высоты
            estimated_lines = max(2, (original_line[1] - original_line[0]) // min_line_height)
            
            # Ищем минимумы в проекции для разделения
            new_lines = []
            segment_height = (original_line[1] - original_line[0]) // estimated_lines
            
            for i in range(estimated_lines):
                seg_start = original_line[0] + i * segment_height
                seg_end = original_line[0] + (i + 1) * segment_height if i < estimated_lines - 1 else original_line[1]
                
                # В каждом сегменте ищем локальный минимум как точку разделения
                if i > 0:  # Не для первого сегмента
                    seg_proj = line_projection[seg_start - original_line[0]:seg_end - original_line[0]]
                    if len(seg_proj) > 0:
                        # Находим минимум в начале сегмента
                        search_window = min(segment_height // 2, len(seg_proj))
                        if search_window > 0:
                            min_idx = np.argmin(seg_proj[:search_window])
                            split_point = seg_start + min_idx
                            # Обновляем конец предыдущей строки и начало текущей
                            if new_lines:
                                new_lines[-1] = (new_lines[-1][0], split_point)
                            seg_start = split_point
                
                if seg_end > seg_start + min_line_height * 0.5:
                    new_lines.append((seg_start, seg_end))
            
            if len(new_lines) > 1:
                print(f"  Разделено на {len(new_lines)} строк")
                lines = new_lines
        
        if single_line_too_large or len(lines) < expected_lines:
            # Ищем локальные минимумы в проекции для разделения строк
            try:
                from scipy.signal import find_peaks
                # Инвертируем проекцию для поиска минимумов
                inverted = max_val - smoothed_projection
                # Находим пики (минимумы в оригинальной проекции)
                peaks, properties = find_peaks(inverted, height=max_val * 0.3, distance=min_line_height)
                
                if len(peaks) > 0:
                    # Разделяем строку по минимумам
                    original_line = lines[0]
                    new_lines = []
                    start = original_line[0]
                    for peak in peaks:
                        # Проверяем, что пик находится внутри исходной строки
                        if original_line[0] < peak < original_line[1]:
                            if peak > start + min_line_height:
                                new_lines.append((start, peak))
                                start = peak
                    if start < original_line[1] - min_line_height:
                        new_lines.append((start, original_line[1]))
                    
                    if len(new_lines) > 1:
                        lines = new_lines
            except ImportError:
                # Fallback: простой поиск локальных минимумов без scipy
                original_line = lines[0] if lines else (0, gray.shape[0])
                line_projection = smoothed_projection[original_line[0]:original_line[1]]
                if len(line_projection) > min_line_height * 2:
                    # Ищем локальные минимумы вручную - более агрессивно
                    min_val = np.min(line_projection)
                    max_val_line = np.max(line_projection)
                    # Более низкий порог для поиска минимумов
                    threshold_min = min_val + (max_val_line - min_val) * 0.2
                    
                    # Находим точки, где проекция ниже порога и является локальным минимумом
                    local_mins = []
                    window_size = max(3, min_line_height // 4)  # Окно для поиска минимумов
                    
                    for i in range(min_line_height, len(line_projection) - min_line_height):
                        # Проверяем, является ли точка локальным минимумом в окне
                        is_local_min = True
                        center_val = line_projection[i]
                        
                        # Проверяем, что это минимум в окне
                        for j in range(max(0, i - window_size), min(len(line_projection), i + window_size + 1)):
                            if j != i and line_projection[j] < center_val:
                                is_local_min = False
                                break
                        
                        if is_local_min and center_val < threshold_min:
                            # Проверяем расстояние от предыдущего минимума
                            if not local_mins or (i + original_line[0] - local_mins[-1]) > min_line_height * 0.8:
                                local_mins.append(i + original_line[0])
                    
                    # Если не нашли достаточно минимумов, пробуем еще более агрессивный подход
                    if len(local_mins) < 2:
                        # Ищем все точки ниже медианы как потенциальные разделители
                        median_val = np.median(line_projection)
                        for i in range(min_line_height, len(line_projection) - min_line_height, max(1, min_line_height // 2)):
                            if line_projection[i] < median_val * 0.7:
                                if not local_mins or (i + original_line[0] - local_mins[-1]) > min_line_height * 0.6:
                                    local_mins.append(i + original_line[0])
                    
                    if len(local_mins) > 0:
                        new_lines = []
                        start = original_line[0]
                        for peak in local_mins:
                            if peak > start + min_line_height * 0.8:
                                new_lines.append((start, peak))
                                start = peak
                        if start < original_line[1] - min_line_height * 0.8:
                            new_lines.append((start, original_line[1]))
                        
                        if len(new_lines) > 1:
                            lines = new_lines
                        elif len(new_lines) == 1 and new_lines[0][1] - new_lines[0][0] < original_line[1] - original_line[0]:
                            # Хотя бы немного разделили
                            lines = new_lines
        
        # Финальная проверка: если все еще одна большая строка, принудительно делим на части
        if len(lines) == 1:
            line_height = lines[0][1] - lines[0][0]
            if line_height > gray.shape[0] * 0.15:  # Если строка больше 15% изображения
                print(f"  Финальная проверка: принудительно разделяем строку высотой {line_height}px...")
                # Делим на равные части на основе типичной высоты строки
                typical_height = min_line_height * 2 if min_line_height > 0 else max(20, line_height // 5)
                num_splits = max(2, line_height // typical_height)
                
                new_lines = []
                split_size = line_height // num_splits
                for i in range(num_splits):
                    start = lines[0][0] + i * split_size
                    end = lines[0][0] + (i + 1) * split_size if i < num_splits - 1 else lines[0][1]
                    if end - start >= typical_height * 0.5:
                        new_lines.append((start, end))
                
                if len(new_lines) > 1:
                    print(f"  Принудительно разделено на {len(new_lines)} строк")
                    lines = new_lines
        
        # Извлекаем изображения строк
        line_images = []
        for line_start, line_end in lines:
            # Добавляем небольшой отступ
            padding = max(3, int(min_line_height * 0.2))
            y0 = max(0, line_start - padding)
            y1 = min(gray.shape[0], line_end + padding)
            
            if y1 > y0 and (y1 - y0) >= min_line_height:
                # Обрезаем строку (берем всю ширину изображения)
                line_img = gray[y0:y1, :]
                
                # Удаляем пустые столбцы слева и справа
                # Вертикальная проекция для удаления пустых краев
                vertical_projection = np.sum(line_img, axis=0)
                if np.max(vertical_projection) > 0:
                    threshold_v = np.max(vertical_projection) * 0.05
                    
                    # Находим начало и конец текста
                    x0 = 0
                    x1 = line_img.shape[1]
                    for i, value in enumerate(vertical_projection):
                        if value > threshold_v:
                            x0 = max(0, i - padding)
                            break
                    
                    for i in range(len(vertical_projection) - 1, -1, -1):
                        if vertical_projection[i] > threshold_v:
                            x1 = min(line_img.shape[1], i + padding)
                            break
                    
                    if x1 > x0:
                        line_img = line_img[:, x0:x1]
                        # Конвертируем в PIL Image
                        line_pil = Image.fromarray(line_img).convert('RGB')
                        line_images.append(line_pil)
        
        return line_images if line_images else None
        
    except Exception as e:
        print(f"  Ошибка при сегментации с проекцией: {e}")
        import traceback
        traceback.print_exc()
        return None


def segment_with_paddleocr(image_path):
    """
    Сегментация изображения на строки с использованием PaddleOCR
    
    Args:
        image_path: путь к изображению
        
    Returns:
        список PIL Image объектов - изображения отдельных строк
    """
    try:
        from paddleocr import PaddleOCR
        
        # Инициализируем PaddleOCR (только для детекции, без распознавания)
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Загружаем изображение
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        # Получаем результаты детекции (bounding boxes текстовых регионов)
        result = ocr.ocr(img_array, cls=True)
        
        if not result or not result[0]:
            return None
        
        # Группируем детекции по строкам (по Y координате)
        detections = result[0]
        line_boxes = []
        
        for detection in detections:
            if detection:
                box = detection[0]  # Координаты bounding box
                # box имеет формат [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                if len(box) >= 4:
                    # Находим минимальные и максимальные координаты
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    x0, x1 = int(min(x_coords)), int(max(x_coords))
                    y0, y1 = int(min(y_coords)), int(max(y_coords))
                    line_boxes.append((x0, y0, x1, y1))
        
        if not line_boxes:
            return None
        
        # Сортируем по Y координате (сверху вниз)
        line_boxes.sort(key=lambda b: b[1])
        
        # Группируем близкие по Y координате боксы в одну строку
        grouped_lines = []
        current_line = [line_boxes[0]]
        
        for box in line_boxes[1:]:
            # Если Y координата близка к текущей строке, добавляем в неё
            avg_y_current = sum(b[1] for b in current_line) / len(current_line)
            if abs(box[1] - avg_y_current) < 30:  # Порог для группировки
                current_line.append(box)
            else:
                # Новая строка
                grouped_lines.append(current_line)
                current_line = [box]
        grouped_lines.append(current_line)
        
        # Извлекаем изображения строк
        line_images = []
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        for line_group in grouped_lines:
            # Находим общий bounding box для группы
            x0 = min(b[0] for b in line_group)
            y0 = min(b[1] for b in line_group)
            x1 = max(b[2] for b in line_group)
            y1 = max(b[3] for b in line_group)
            
            # Добавляем отступ
            padding = 5
            x0 = max(0, x0 - padding)
            y0 = max(0, y0 - padding)
            x1 = min(gray.shape[1], x1 + padding)
            y1 = min(gray.shape[0], y1 + padding)
            
            if x1 > x0 and y1 > y0:
                line_img = gray[y0:y1, x0:x1]
                line_pil = Image.fromarray(line_img).convert('RGB')
                line_images.append(line_pil)
        
        return line_images if line_images else None
        
    except ImportError:
        return None
    except Exception as e:
        print(f"  Ошибка при сегментации с PaddleOCR: {e}")
        return None


def segment_with_easyocr(image_path):
    """
    Сегментация изображения на строки с использованием EasyOCR
    
    Args:
        image_path: путь к изображению
        
    Returns:
        список PIL Image объектов - изображения отдельных строк
    """
    try:
        import easyocr
        
        # Инициализируем EasyOCR (только для детекции)
        reader = easyocr.Reader(['en', 'ru'], gpu=False)
        
        # Загружаем изображение
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        # Получаем результаты детекции
        results = reader.readtext(img_array, paragraph=False)
        
        if not results:
            return None
        
        # Группируем детекции по строкам
        line_boxes = []
        for result in results:
            bbox = result[0]  # Координаты bounding box
            # bbox имеет формат [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            if len(bbox) >= 4:
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x0, x1 = int(min(x_coords)), int(max(x_coords))
                y0, y1 = int(min(y_coords)), int(max(y_coords))
                line_boxes.append((x0, y0, x1, y1))
        
        if not line_boxes:
            return None
        
        # Сортируем и группируем по Y координате
        line_boxes.sort(key=lambda b: b[1])
        grouped_lines = []
        current_line = [line_boxes[0]]
        
        for box in line_boxes[1:]:
            avg_y_current = sum(b[1] for b in current_line) / len(current_line)
            if abs(box[1] - avg_y_current) < 30:
                current_line.append(box)
            else:
                grouped_lines.append(current_line)
                current_line = [box]
        grouped_lines.append(current_line)
        
        # Извлекаем изображения строк
        line_images = []
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        for line_group in grouped_lines:
            x0 = min(b[0] for b in line_group)
            y0 = min(b[1] for b in line_group)
            x1 = max(b[2] for b in line_group)
            y1 = max(b[3] for b in line_group)
            
            padding = 5
            x0 = max(0, x0 - padding)
            y0 = max(0, y0 - padding)
            x1 = min(gray.shape[1], x1 + padding)
            y1 = min(gray.shape[0], y1 + padding)
            
            if x1 > x0 and y1 > y0:
                line_img = gray[y0:y1, x0:x1]
                line_pil = Image.fromarray(line_img).convert('RGB')
                line_images.append(line_pil)
        
        return line_images if line_images else None
        
    except ImportError:
        return None
    except Exception as e:
        print(f"  Ошибка при сегментации с EasyOCR: {e}")
        return None


# Глобальный кэш для модели YOLOv8 (чтобы не загружать каждый раз)
_yolov8_model_cache = None

def segment_with_yolov8(image_path, conf_threshold=0.15, iou_threshold=0.45, max_detections=500):
    """
    Сегментация изображения на строки с использованием YOLOv8 для детекции рукописного текста
    Пытается использовать модель armvectores/yolov8n_handwritten_text_detection
    Если модель недоступна, возвращает None (будет использован следующий метод)
    
    Args:
        image_path: путь к изображению
        conf_threshold: порог уверенности для детекции (по умолчанию 0.25)
        iou_threshold: порог IoU для NMS (по умолчанию 0.45)
        max_detections: максимальное количество детекций (по умолчанию 300)
        
    Returns:
        список PIL Image объектов - изображения отдельных строк или None если модель недоступна
    """
    global _yolov8_model_cache
    
    try:
        from ultralytics import YOLO
        import torch
        
        # Загружаем изображение
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        img_height, img_width = img_array.shape[:2]
        
        # Кэшируем модель, чтобы не загружать каждый раз
        if _yolov8_model_cache is None:
            from huggingface_hub import hf_hub_download
            
            repo_id = "armvectores/yolov8n_handwritten_text_detection"
            filename = "best.pt"
            
            try:
                print(f"  Загрузка модели YOLOv8 из Hugging Face: {repo_id}/{filename}...")
                model_path = hf_hub_download(
                    local_dir=".",
                    repo_id=repo_id,
                    filename=filename
                )
                _yolov8_model_cache = YOLO(model_path)
                
                # Определяем устройство (GPU/CPU)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _yolov8_model_cache.to(device)
                print(f"  Модель успешно загружена: {model_path} (устройство: {device})")
            except Exception as e:
                print(f"  Ошибка загрузки модели YOLOv8: {e}")
                print("  Используйте другой метод сегментации или проверьте доступность модели")
                return None
        
        model = _yolov8_model_cache
        
        # Адаптивный порог уверенности в зависимости от размера изображения
        # Для больших изображений используем более низкий порог
        if img_height > 2000 or img_width > 2000:
            conf_threshold = max(0.1, conf_threshold * 0.7)
            print(f"  Большое изображение ({img_width}x{img_height}), используем порог уверенности: {conf_threshold:.2f}")
        
        # Для многострочных изображений снижаем порог еще больше
        # Оцениваем количество строк на основе высоты изображения
        estimated_lines = max(1, img_height // 50)  # Примерно 50px на строку
        if estimated_lines > 15:
            conf_threshold = max(0.1, conf_threshold * 0.9)
            print(f"  Многострочное изображение (оценка: ~{estimated_lines} строк), порог уверенности: {conf_threshold:.2f}")
        
        # Выполняем детекцию с улучшенными параметрами
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_detections,
            verbose=False,
            agnostic_nms=False,  # Не группировать классы вместе
            retina_masks=False
        )
        
        if not results or len(results) == 0:
            return None
        
        # Извлекаем bounding boxes из результатов
        line_boxes = []
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # Координаты в формате [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()
            
            # Минимальные размеры для фильтрации (адаптивные к размеру изображения)
            min_width = max(10, int(img_width * 0.01))
            min_height = max(10, int(img_height * 0.01))
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                width = x2 - x1
                height = y2 - y1
                
                # Фильтруем слишком маленькие детекции и проверяем соотношение сторон
                aspect_ratio = width / height if height > 0 else 0
                
                # Текст обычно имеет ширину больше высоты (aspect_ratio > 0.5)
                if (width >= min_width and height >= min_height and 
                    aspect_ratio > 0.3 and aspect_ratio < 20):  # Разумные пределы
                    line_boxes.append((x1, y1, x2, y2, conf))  # Сохраняем также уверенность
        
        if not line_boxes:
            return None
        
        # Сортируем по Y координате (сверху вниз), затем по X (слева направо)
        line_boxes.sort(key=lambda b: (b[1], b[0]))
        
        # Улучшенная группировка близких по Y координате боксов в строки
        grouped_lines = []
        if line_boxes:
            current_line = [line_boxes[0]]
            
            for box in line_boxes[1:]:
                # Вычисляем среднюю Y координату и высоту текущей строки
                avg_y_current = sum(b[1] for b in current_line) / len(current_line)
                avg_height_current = sum(b[3] - b[1] for b in current_line) / len(current_line)
                # Уверенность хранится в 5-м элементе (индекс 4)
                avg_confidence = sum(b[4] for b in current_line) / len(current_line) if len(current_line[0]) > 4 else 0.5
                
                # Адаптивный порог группировки на основе высоты строки
                # Используем более строгий порог для строк с высокой уверенностью
                # Но делаем его более консервативным, чтобы не объединять разные строки
                grouping_threshold = avg_height_current * (0.4 if avg_confidence > 0.5 else 0.5)
                
                # Проверяем перекрытие по Y координате
                box_y_center = (box[1] + box[3]) / 2
                box_height = box[3] - box[1]
                overlap = abs(box_y_center - avg_y_current)
                
                # Также проверяем, не слишком ли далеко по X (чтобы не группировать далекие слова)
                max_x_current = max(b[2] for b in current_line)
                max_gap_x = img_width * 0.15  # Увеличиваем до 15% ширины изображения
                
                # Дополнительная проверка: если высота нового бокса сильно отличается,
                # это может быть другая строка
                height_ratio = box_height / avg_height_current if avg_height_current > 0 else 1.0
                height_similar = 0.5 < height_ratio < 2.0  # Высота должна быть в пределах 50%-200%
                
                # Более строгая проверка для предотвращения объединения разных строк
                if (overlap < grouping_threshold and 
                    (box[0] - max_x_current) < max_gap_x and
                    height_similar):
                    current_line.append(box)
                else:
                    # Новая строка
                    grouped_lines.append(current_line)
                    current_line = [box]
            grouped_lines.append(current_line)
        
        # Фильтруем строки с слишком маленьким количеством детекций (возможно, шум)
        min_detections_per_line = 1  # Минимум 1 детекция на строку
        filtered_grouped_lines = [line for line in grouped_lines if len(line) >= min_detections_per_line]
        
        if not filtered_grouped_lines:
            return None
        
        # Проверяем, не слишком ли мало строк найдено
        # Если ожидается много строк, но найдено мало, пробуем более агрессивную группировку
        if len(filtered_grouped_lines) < estimated_lines * 0.5 and estimated_lines > 10:
            print(f"  Найдено {len(filtered_grouped_lines)} строк, ожидается ~{estimated_lines}, пробуем более агрессивную группировку...")
            # Перегруппируем с более строгими параметрами
            filtered_grouped_lines = []
            current_line = [line_boxes[0]]
            
            for box in line_boxes[1:]:
                avg_y_current = sum(b[1] for b in current_line) / len(current_line)
                avg_height_current = sum(b[3] - b[1] for b in current_line) / len(current_line)
                box_y_center = (box[1] + box[3]) / 2
                box_height = box[3] - box[1]
                
                # Более строгий порог группировки
                grouping_threshold = avg_height_current * 0.3
                height_ratio = box_height / avg_height_current if avg_height_current > 0 else 1.0
                height_similar = 0.4 < height_ratio < 2.5
                
                max_x_current = max(b[2] for b in current_line)
                max_gap_x = img_width * 0.2
                
                if (abs(box_y_center - avg_y_current) < grouping_threshold and 
                    (box[0] - max_x_current) < max_gap_x and
                    height_similar):
                    current_line.append(box)
                else:
                    filtered_grouped_lines.append(current_line)
                    current_line = [box]
            filtered_grouped_lines.append(current_line)
            
            print(f"  После перегруппировки найдено {len(filtered_grouped_lines)} строк")
        
        # Извлекаем изображения строк
        line_images = []
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        for line_group in filtered_grouped_lines:
            # Находим общий bounding box для группы
            x0 = min(b[0] for b in line_group)
            y0 = min(b[1] for b in line_group)
            x1 = max(b[2] for b in line_group)
            y1 = max(b[3] for b in line_group)
            
            # Адаптивный отступ в зависимости от размера строки
            line_height = y1 - y0
            padding = max(3, int(line_height * 0.1))  # 10% от высоты строки
            
            x0 = max(0, x0 - padding)
            y0 = max(0, y0 - padding)
            x1 = min(gray.shape[1], x1 + padding)
            y1 = min(gray.shape[0], y1 + padding)
            
            if x1 > x0 and y1 > y0:
                line_img = gray[y0:y1, x0:x1]
                
                # Улучшаем контраст для лучшего распознавания
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                line_img = clahe.apply(line_img)
                
                line_pil = Image.fromarray(line_img).convert('RGB')
                line_images.append(line_pil)
        
        return line_images if line_images else None
        
    except ImportError as e:
        print(f"  YOLOv8 недоступен: {e}")
        print("  Установите: pip install ultralytics")
        return None
    except Exception as e:
        print(f"  Ошибка при сегментации с YOLOv8: {e}")
        import traceback
        traceback.print_exc()
        return None


def segment_with_contours(image_path):
    """
    Сегментация изображения на строки с использованием контуров OpenCV
    Более продвинутый метод, чем простая проекция
    
    Args:
        image_path: путь к изображению
        
    Returns:
        список PIL Image объектов - изображения отдельных строк
    """
    try:
        # Загружаем изображение
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        # Конвертируем в grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Улучшаем контраст
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Морфологические операции для соединения символов
        kernel_width = max(30, int(gray.shape[1] * 0.1))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        dilated = cv2.dilate(binary, horizontal_kernel, iterations=1)
        
        # Находим контуры
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Получаем bounding boxes для контуров
        line_boxes = []
        min_line_height = max(10, int(gray.shape[0] * 0.015))  # Уменьшено для лучшего обнаружения
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h >= min_line_height:  # Фильтруем слишком маленькие контуры
                line_boxes.append((x, y, x + w, y + h))
        
        if not line_boxes:
            return None
        
        # Сортируем по Y координате
        line_boxes.sort(key=lambda b: b[1])
        
        # Группируем близкие по Y координате боксы
        grouped_lines = []
        current_line = [line_boxes[0]]
        
        for box in line_boxes[1:]:
            avg_y_current = sum(b[1] for b in current_line) / len(current_line)
            avg_height_current = sum(b[3] - b[1] for b in current_line) / len(current_line)
            # Более агрессивное разделение - используем меньший порог
            if abs(box[1] - avg_y_current) < avg_height_current * 0.6:  # 60% от средней высоты
                current_line.append(box)
            else:
                grouped_lines.append(current_line)
                current_line = [box]
        grouped_lines.append(current_line)
        
        # Если получилось слишком мало строк, пробуем более агрессивное разделение
        if len(grouped_lines) < 2 and len(line_boxes) > 1:
            # Перегруппируем с меньшим порогом
            grouped_lines = []
            current_line = [line_boxes[0]]
            for box in line_boxes[1:]:
                avg_y_current = sum(b[1] for b in current_line) / len(current_line)
                avg_height_current = sum(b[3] - b[1] for b in current_line) / len(current_line)
                if abs(box[1] - avg_y_current) < avg_height_current * 0.4:  # Еще более агрессивно
                    current_line.append(box)
                else:
                    grouped_lines.append(current_line)
                    current_line = [box]
            grouped_lines.append(current_line)
        
        # Извлекаем изображения строк
        line_images = []
        for line_group in grouped_lines:
            x0 = min(b[0] for b in line_group)
            y0 = min(b[1] for b in line_group)
            x1 = max(b[2] for b in line_group)
            y1 = max(b[3] for b in line_group)
            
            padding = 5
            x0 = max(0, x0 - padding)
            y0 = max(0, y0 - padding)
            x1 = min(gray.shape[1], x1 + padding)
            y1 = min(gray.shape[0], y1 + padding)
            
            if x1 > x0 and y1 > y0:
                line_img = gray[y0:y1, x0:x1]
                line_pil = Image.fromarray(line_img).convert('RGB')
                line_images.append(line_pil)
        
        return line_images if line_images else None
        
    except Exception as e:
        print(f"  Ошибка при сегментации с контурами: {e}")
        import traceback
        traceback.print_exc()
        return None


def segment_image_to_lines(image_path, method='auto'):
    """
    Основная функция для сегментации изображения на строки
    Поддерживает несколько методов сегментации
    
    Args:
        image_path: путь к изображению
        method: метод сегментации ('auto', 'kraken', 'paddleocr', 'easyocr', 'contours', 'projection')
                'auto' - пробует методы в порядке приоритета
        
    Returns:
        список PIL Image объектов - изображения отдельных строк
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    
    line_images = None
    
    if method == 'auto':
        # Пробуем методы в порядке приоритета
        methods = [
            ('YOLOv8', segment_with_yolov8),
            ('PaddleOCR', segment_with_paddleocr),
            ('EasyOCR', segment_with_easyocr),
            ('Kraken', segment_with_kraken),
            ('контуры', segment_with_contours),
            ('проекция', segment_with_projection),
        ]
        
        for method_name, segment_func in methods:
            try:
                print(f"  Пробуем сегментацию с {method_name}...")
                line_images = segment_func(image_path)
                if line_images and len(line_images) > 0:
                    print(f"  Успешно использован метод: {method_name}")
                    break
            except Exception as e:
                print(f"  {method_name} недоступен: {e}")
                continue
    else:
        # Используем указанный метод
        method_map = {
            'yolov8': segment_with_yolov8,
            'kraken': segment_with_kraken,
            'paddleocr': segment_with_paddleocr,
            'easyocr': segment_with_easyocr,
            'contours': segment_with_contours,
            'projection': segment_with_projection,
        }
        
        if method.lower() in method_map:
            print(f"  Используем метод сегментации: {method}")
            line_images = method_map[method.lower()](image_path)
        else:
            print(f"  Неизвестный метод: {method}, используем auto")
            return segment_image_to_lines(image_path, method='auto')
    
    if not line_images or len(line_images) == 0:
        print("  Предупреждение: не удалось сегментировать изображение на строки")
        return []
    
    # Фильтруем слишком маленькие строки
    filtered_lines = []
    for line_img in line_images:
        width, height = line_img.size
        if width >= 10 and height >= 10:  # Минимальный размер строки
            filtered_lines.append(line_img)
    
    return filtered_lines


if __name__ == "__main__":
    """
    Тестирование модуля сегментации
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Тестирование сегментации изображений на строки")
    parser.add_argument("--image", type=str, required=True, help="Путь к изображению")
    parser.add_argument("--output", type=str, default="test_segmentation", 
                       help="Папка для сохранения сегментированных строк")
    parser.add_argument("--method", type=str, default="auto",
                       choices=["auto", "yolov8", "kraken", "paddleocr", "easyocr", "contours", "projection"],
                       help="Метод сегментации (по умолчанию: auto)")
    
    args = parser.parse_args()
    
    print(f"Сегментация изображения: {args.image}")
    print(f"Метод: {args.method}")
    line_images = segment_image_to_lines(args.image, method=args.method)
    
    if line_images:
        print(f"Найдено строк: {len(line_images)}")
        
        # Создаем папку для сохранения
        os.makedirs(args.output, exist_ok=True)
        
        # Сохраняем каждую строку
        for i, line_img in enumerate(line_images, 1):
            output_path = os.path.join(args.output, f"line_{i:04d}.png")
            line_img.save(output_path, "PNG")
            print(f"  Сохранена строка {i}: {output_path} (размер: {line_img.size})")
    else:
        print("Не удалось сегментировать изображение на строки")

