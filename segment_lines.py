"""
Модуль для сегментации изображений рукописного текста на отдельные строки
Использует YOLOv8 с моделью armvectores/yolov8n_handwritten_text_detection для детекции текста
"""
import os
import cv2
import numpy as np
from PIL import Image


# Глобальный кэш для модели YOLOv8 (чтобы не загружать каждый раз)
_yolov8_model_cache = None

def segment_with_yolov8(image_path, conf_threshold=0.1, iou_threshold=0.5, max_detections=1000):
    """
    Сегментация изображения на строки с использованием YOLOv8 для детекции рукописного текста
    Пытается использовать модель armvectores/yolov8n_handwritten_text_detection
    
    Примечание: Модель детектирует отдельные слова, которые затем группируются в строки
    с помощью улучшенного алгоритма группировки.
    
    Args:
        image_path: путь к изображению
        conf_threshold: порог уверенности для детекции (по умолчанию 0.1)
        iou_threshold: порог IoU для NMS (по умолчанию 0.5)
        max_detections: максимальное количество детекций (по умолчанию 1000)
        
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
                print("  Проверьте доступность модели")
                return None
        
        model = _yolov8_model_cache
        
        # Адаптивный порог уверенности в зависимости от размера изображения
        # Для больших изображений используем более низкий порог
        if img_height > 2000 or img_width > 2000:
            conf_threshold = max(0.03, conf_threshold * 0.5)  # Еще более агрессивное снижение
            print(f"  Большое изображение ({img_width}x{img_height}), используем порог уверенности: {conf_threshold:.2f}")
        
        # Для многострочных изображений снижаем порог еще больше
        # Оцениваем количество строк на основе высоты изображения
        estimated_lines = max(1, img_height // 50)  # Примерно 50px на строку
        if estimated_lines > 10:
            conf_threshold = max(0.03, conf_threshold * 0.7)  # Более агрессивное снижение
            print(f"  Многострочное изображение (оценка: ~{estimated_lines} строк), порог уверенности: {conf_threshold:.2f}")
        
        # Для рукописного текста используем еще более низкий порог
        # Минимальный порог снижен до 0.03 для поиска всех слов
        conf_threshold = max(0.03, conf_threshold * 0.85)
        print(f"  Используем порог уверенности для рукописного текста: {conf_threshold:.2f}")
        
        # Выполняем детекцию с улучшенными параметрами
        # Используем более мягкие параметры для поиска всех слов
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_detections,
            verbose=False,
            agnostic_nms=False,  # Не группировать классы вместе
            retina_masks=False,
            # Дополнительные параметры для лучшего поиска
            save=False,  # Не сохранять изображения
            show=False   # Не показывать результаты
        )
        
        print(f"  YOLOv8 выполнил детекцию с порогом уверенности: {conf_threshold:.3f}")
        
        if not results or len(results) == 0:
            return None
        
        # Извлекаем bounding boxes из результатов
        line_boxes = []
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # Координаты в формате [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()
            
            # Минимальные размеры для фильтрации (адаптивные к размеру изображения)
            # Очень мягкие ограничения для рукописного текста - принимаем почти все детекции
            min_width = max(3, int(img_width * 0.003))  # Еще более мягкие ограничения
            min_height = max(3, int(img_height * 0.003))  # Еще более мягкие ограничения
            
            print(f"  Всего детекций от модели: {len(boxes)}")
            print(f"  Фильтруем детекции (мин. размер: {min_width}x{min_height})...")
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                width = x2 - x1
                height = y2 - y1
                
                # Фильтруем слишком маленькие детекции и проверяем соотношение сторон
                aspect_ratio = width / height if height > 0 else 0
                
                # Для рукописного текста принимаем очень широкий диапазон соотношений сторон
                # (отдельные буквы могут быть узкими, слова - широкими, строки - очень широкими)
                # Расширяем пределы еще больше для поиска всех слов
                if (width >= min_width and height >= min_height and 
                    aspect_ratio > 0.1 and aspect_ratio < 50):  # Очень широкие пределы
                    line_boxes.append((x1, y1, x2, y2, conf))  # Сохраняем также уверенность
            
            print(f"  После фильтрации осталось {len(line_boxes)} детекций (из {len(boxes)})")
        
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
                # Для рукописного текста используем более мягкий порог, чтобы объединять слова в строки
                # Увеличиваем порог для лучшего объединения слов
                grouping_threshold = avg_height_current * (0.8 if avg_confidence > 0.3 else 1.0)
                
                # Проверяем перекрытие по Y координате
                box_y_center = (box[1] + box[3]) / 2
                box_height = box[3] - box[1]
                overlap = abs(box_y_center - avg_y_current)
                
                # Также проверяем, не слишком ли далеко по X (чтобы не группировать далекие слова)
                max_x_current = max(b[2] for b in current_line)
                min_x_current = min(b[0] for b in current_line)
                # Увеличиваем максимальный разрыв по X для объединения слов в строки
                # Используем более агрессивное объединение - до 35% ширины изображения
                max_gap_x = img_width * 0.35  # Увеличено до 35% ширины изображения
                
                # Дополнительная проверка: если высота нового бокса сильно отличается,
                # это может быть другая строка (но для рукописного текста делаем более мягко)
                height_ratio = box_height / avg_height_current if avg_height_current > 0 else 1.0
                height_similar = 0.2 < height_ratio < 5.0  # Еще более широкие пределы: 20%-500%
                
                # Проверяем перекрытие по Y более тщательно
                # Используем не только центр, но и проверяем перекрытие bounding boxes
                box_y_top = box[1]
                box_y_bottom = box[3]
                line_y_top = min(b[1] for b in current_line)
                line_y_bottom = max(b[3] for b in current_line)
                
                # Проверяем, есть ли перекрытие по Y между боксами
                y_overlap = max(0, min(box_y_bottom, line_y_bottom) - max(box_y_top, line_y_top))
                overlap_ratio = y_overlap / min(box_height, avg_height_current) if min(box_height, avg_height_current) > 0 else 0
                
                # Более мягкая проверка для объединения слов в строки
                # Группируем, если:
                # 1. Y координата близка (overlap < threshold ИЛИ есть перекрытие по Y)
                # 2. Не слишком далеко по X
                # 3. Высота похожа (или игнорируем проверку высоты для лучшего объединения)
                # Снижаем требования к перекрытию для лучшего объединения
                if ((overlap < grouping_threshold or overlap_ratio > 0.15) and 
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
        
        # Проверяем, не слишком ли много отдельных детекций (слов) вместо строк
        # Если среднее количество детекций на строку слишком маленькое, значит слова не группируются
        avg_detections_per_line = len(line_boxes) / len(filtered_grouped_lines) if filtered_grouped_lines else 0
        
        # Если среднее количество детекций на строку меньше 3, пробуем более агрессивную группировку
        if avg_detections_per_line < 3 and len(line_boxes) > len(filtered_grouped_lines):
            print(f"  Среднее количество детекций на строку: {avg_detections_per_line:.1f}, пробуем более агрессивную группировку...")
            # Перегруппируем с более мягкими параметрами для объединения слов
            filtered_grouped_lines = []
            current_line = [line_boxes[0]]
            
            for box in line_boxes[1:]:
                avg_y_current = sum(b[1] for b in current_line) / len(current_line)
                avg_height_current = sum(b[3] - b[1] for b in current_line) / len(current_line)
                box_y_center = (box[1] + box[3]) / 2
                box_height = box[3] - box[1]
                
                # Очень мягкий порог группировки для объединения слов в строки
                grouping_threshold = avg_height_current * 1.2  # Увеличено до 1.2 для максимального объединения
                height_ratio = box_height / avg_height_current if avg_height_current > 0 else 1.0
                height_similar = 0.15 < height_ratio < 6.0  # Очень широкие пределы
                
                max_x_current = max(b[2] for b in current_line)
                max_gap_x = img_width * 0.4  # Увеличено до 40% для объединения далеких слов
                
                # Проверяем перекрытие по Y
                box_y_top = box[1]
                box_y_bottom = box[3]
                line_y_top = min(b[1] for b in current_line)
                line_y_bottom = max(b[3] for b in current_line)
                y_overlap = max(0, min(box_y_bottom, line_y_bottom) - max(box_y_top, line_y_top))
                overlap_ratio = y_overlap / min(box_height, avg_height_current) if min(box_height, avg_height_current) > 0 else 0
                
                if ((abs(box_y_center - avg_y_current) < grouping_threshold or overlap_ratio > 0.2) and 
                    (box[0] - max_x_current) < max_gap_x and
                    height_similar):
                    current_line.append(box)
                else:
                    filtered_grouped_lines.append(current_line)
                    current_line = [box]
            filtered_grouped_lines.append(current_line)
            
            print(f"  После перегруппировки найдено {len(filtered_grouped_lines)} строк (было {len(grouped_lines)})")
        
        # Дополнительная проверка: если все еще слишком мало строк найдено
        if len(filtered_grouped_lines) < estimated_lines * 0.5 and estimated_lines > 10:
            print(f"  Найдено {len(filtered_grouped_lines)} строк, ожидается ~{estimated_lines}, пробуем еще более агрессивную группировку...")
            # Используем еще более мягкие параметры
            filtered_grouped_lines = []
            current_line = [line_boxes[0]]
            
            for box in line_boxes[1:]:
                avg_y_current = sum(b[1] for b in current_line) / len(current_line)
                avg_height_current = sum(b[3] - b[1] for b in current_line) / len(current_line)
                box_y_center = (box[1] + box[3]) / 2
                
                # Очень мягкий порог - группируем почти все, что близко по Y
                grouping_threshold = avg_height_current * 1.2
                max_x_current = max(b[2] for b in current_line)
                max_gap_x = img_width * 0.4
                
                if (abs(box_y_center - avg_y_current) < grouping_threshold and 
                    (box[0] - max_x_current) < max_gap_x):
                    current_line.append(box)
                else:
                    filtered_grouped_lines.append(current_line)
                    current_line = [box]
            filtered_grouped_lines.append(current_line)
            
            print(f"  После финальной перегруппировки найдено {len(filtered_grouped_lines)} строк")
        
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


def segment_image_to_lines(image_path, method='yolov8'):
    """
    Основная функция для сегментации изображения на строки
    Использует YOLOv8 для детекции текста
    
    Args:
        image_path: путь к изображению
        method: метод сегментации (по умолчанию 'yolov8', другие методы удалены)
        
    Returns:
        список PIL Image объектов - изображения отдельных строк
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    
    # Используем только YOLOv8
    if method.lower() != 'yolov8':
        print(f"  Предупреждение: метод '{method}' не поддерживается, используем YOLOv8")
    
    print(f"  Используем метод сегментации: YOLOv8")
    line_images = segment_with_yolov8(image_path)
    
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
    parser.add_argument("--method", type=str, default="yolov8",
                       choices=["yolov8"],
                       help="Метод сегментации (по умолчанию: yolov8)")
    
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
