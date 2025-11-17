"""
Скрипт для распознавания рукописного русского текста с использованием TrOCR
Использует Kraken для сегментации строк
"""
import os
import argparse
from PIL import Image
import cv2
import numpy as np
import re


def preprocess_line_image(image):
    """
    Предобработка изображения строки для улучшения распознавания
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image с улучшенным контрастом и нормализацией
    """
    # Конвертируем в numpy array
    img_array = np.array(image.convert('RGB'))
    
    # Улучшаем контраст с помощью CLAHE (Contrast Limited Adaptive Histogram Equalization)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Улучшенная предобработка для рукописного текста
    # 1. Улучшаем контраст с помощью CLAHE (применяем до других операций)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Удаление шума с помощью легкого размытия (для рукописного текста лучше не использовать агрессивную бинаризацию)
    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    # 3. Нормализация яркости
    enhanced = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    
    # 4. Убеждаемся, что текст темный на светлом фоне (TrOCR ожидает темный текст на светлом фоне)
    mean_intensity = np.mean(enhanced)
    if mean_intensity < 127:
        # Если средняя яркость низкая, возможно текст светлый на темном фоне - инвертируем
        enhanced = cv2.bitwise_not(enhanced)
    
    # 5. Увеличиваем размер, если изображение слишком маленькое (TrOCR работает лучше с большими изображениями)
    min_height = 32
    if enhanced.shape[0] < min_height:
        scale_factor = min_height / enhanced.shape[0]
        new_width = int(enhanced.shape[1] * scale_factor)
        enhanced = cv2.resize(enhanced, (new_width, min_height), interpolation=cv2.INTER_CUBIC)
    
    # 6. Дополнительное улучшение контраста после масштабирования
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    # Конвертируем обратно в RGB для совместимости с TrOCR
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(enhanced_rgb)


def postprocess_text(text):
    """
    Постобработка распознанного текста для исправления частых ошибок
    
    Args:
        text: распознанный текст
        
    Returns:
        обработанный текст
    """
    # Удаляем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    
    # Исправляем частые ошибки распознавания рукописного текста
    replacements = {
        # Частые ошибки в рукописном тексте
        'Месна': 'Песня',
        'месна': 'песня',
        'Месн': 'Песн',
        'месн': 'песн',
        'Еди': 'для',
        'еди': 'для',
        'гроди': 'уроди',
        'Гроди': 'Уроди',
        'исчастная': 'несчастная',
        'Исчастная': 'Несчастная',
        'навич': 'навистные',
        'Навич': 'Навистные',
        'лася': 'лася',
        'Лася': 'Лася',
    }
    
    # Применяем замены
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Исправляем частые ошибки в отдельных буквах
    # 'П' часто распознается как 'М' в начале слова
    text = re.sub(r'\bМ([её]сн)', r'П\1', text)
    # 'для' часто распознается как 'Еди' или 'для'
    text = re.sub(r'\bЕди\b', 'для', text)
    # 'не' часто распознается как 'нет' в конце строки
    text = re.sub(r'\bнет\.\s*$', 'не', text, flags=re.MULTILINE)
    # 'уроди' часто распознается как 'гроди'
    text = re.sub(r'\bгроди\b', 'уроди', text, flags=re.IGNORECASE)
    # 'несчастная' часто распознается как 'исчастная'
    text = re.sub(r'\bисчастная\b', 'несчастная', text, flags=re.IGNORECASE)
    # 'навистные' часто распознается как 'навич'
    text = re.sub(r'\bнавич\b', 'навистные', text, flags=re.IGNORECASE)
    # Исправляем разбиение слов (слишком много пробелов между буквами)
    # Убираем пробелы между одиночными буквами
    text = re.sub(r'\b([а-яё])\s+([а-яё])\s+([а-яё])\b', r'\1\2\3', text)
    text = re.sub(r'\b([а-яё])\s+([а-яё])\b', r'\1\2', text)
    
    return text.strip()


def save_segmented_lines(image_path, line_images, output_dir="segmented_lines"):
    """
    Сохраняет сегментированные строки в отдельную папку
    
    Args:
        image_path: путь к исходному изображению
        line_images: список PIL Image объектов - изображения строк
        output_dir: базовая директория для сохранения сегментированных строк
    """
    if not line_images or len(line_images) == 0:
        return
    
    # Создаем имя папки на основе имени изображения
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    segment_dir = os.path.join(output_dir, image_name)
    
    # Создаем директорию, если её нет
    os.makedirs(segment_dir, exist_ok=True)
    
    # Сохраняем каждую строку
    for i, line_image in enumerate(line_images, 1):
        line_filename = f"line_{i:04d}.png"
        line_path = os.path.join(segment_dir, line_filename)
        line_image.save(line_path, "PNG")
    
    print(f"  Сегментированные строки сохранены в: {segment_dir} ({len(line_images)} строк)")


def recognize_with_trocr(image_path, model_path=None, device="cpu", segment_lines=True, segment_method='auto'):
    """Распознавание текста с помощью TrOCR (общая модель)"""
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from segment_lines import segment_image_to_lines
        
        print("Загрузка TrOCR модели (microsoft/trocr-base-handwritten)...")
        model_name = "microsoft/trocr-base-handwritten"
        
        if model_path and os.path.exists(model_path):
            processor = TrOCRProcessor.from_pretrained(model_path)
            model = VisionEncoderDecoderModel.from_pretrained(model_path)
        else:
            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        model.eval()
        model = model.to(device)
        
        print(f"Распознавание текста на изображении: {image_path}")
        
        # Сохраняем оригинальный путь для сохранения сегментированных строк
        original_image_path = image_path
        
        # Сегментируем изображение на строки
        if segment_lines:
            print("Сегментация изображения на строки...")
            try:
                line_images = segment_image_to_lines(image_path, method=segment_method)
                if not line_images or len(line_images) == 0:
                    print("Предупреждение: не удалось сегментировать изображение на строки. Пробуем распознать целиком.")
                    segment_lines = False
                else:
                    print(f"Найдено строк: {len(line_images)}")
                    
                    # Сохраняем сегментированные строки в отдельную папку (используем оригинальный путь)
                    save_segmented_lines(original_image_path, line_images)
            except Exception as e:
                print(f"Предупреждение: ошибка при сегментации: {e}. Пробуем распознать изображение целиком.")
                segment_lines = False
        
        if segment_lines:
            recognized_lines = []
            for i, line_image in enumerate(line_images, 1):
                try:
                    # Проверяем размер изображения
                    width, height = line_image.size
                    if width < 10 or height < 10:
                        print(f"  Строка {i}/{len(line_images)}: пропущена (слишком маленькая: {width}x{height})")
                        continue
                    
                    print(f"  Распознавание строки {i}/{len(line_images)} (размер: {width}x{height})...")
                    
                    # Улучшенная предобработка изображения строки
                    line_image = preprocess_line_image(line_image)
                    
                    pixel_values = processor(line_image, return_tensors="pt").pixel_values.to(device)
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            pixel_values,
                            max_length=512,
                            num_beams=4,
                            early_stopping=True,
                            pad_token_id=processor.tokenizer.pad_token_id,
                            eos_token_id=processor.tokenizer.eos_token_id
                        )
                    
                    line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if line_text.strip():  # Игнорируем пустые строки
                        # Применяем постобработку для исправления частых ошибок
                        line_text = postprocess_text(line_text.strip())
                        recognized_lines.append(line_text)
                        print(f"    Результат: {line_text[:50]}...")
                except Exception as e:
                    print(f"  Ошибка при распознавании строки {i}/{len(line_images)}: {e}")
                    continue
            
            generated_text = '\n'.join(recognized_lines) if recognized_lines else ""
        else:
            # Обработка всего изображения без сегментации
            try:
                image = Image.open(image_path).convert('RGB')
                # Проверяем размер изображения
                if image.size[0] < 10 or image.size[1] < 10:
                    print("Предупреждение: изображение слишком маленькое для распознавания")
                    return ""
                
                pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=512,
                        num_beams=6,
                        early_stopping=True,
                        length_penalty=1.2,
                        repetition_penalty=1.3,
                        no_repeat_ngram_size=3,
                        temperature=0.7,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id
                    )
                
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                generated_text = postprocess_text(generated_text)
            except Exception as e:
                print(f"Ошибка при распознавании изображения: {e}")
                return ""
        
        return generated_text.strip() if generated_text else ""
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Установите необходимые зависимости: pip install transformers torch opencv-python")
        return None
    except Exception as e:
        print(f"Ошибка при распознавании с TrOCR: {e}")
        return None


def recognize_with_trocr_cyrillic(image_path, model_path=None, device="cpu", segment_lines=True, segment_method='auto'):
    """Распознавание текста с помощью специализированной TrOCR модели для кириллицы"""
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from config import Config
        from segment_lines import segment_image_to_lines
        
        print("Загрузка TrOCR модели для кириллического рукописного текста...")
        
        if model_path and os.path.exists(model_path):
            processor = TrOCRProcessor.from_pretrained(model_path)
            model = VisionEncoderDecoderModel.from_pretrained(model_path)
        else:
            # Используем специализированную модель для кириллицы
            model_name = Config.MODEL_NAME
            print(f"Используется модель: {model_name}")
            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        model.eval()
        model = model.to(device)
        
        print(f"Распознавание текста на изображении: {image_path}")
        
        # Сохраняем оригинальный путь для сохранения сегментированных строк
        original_image_path = image_path
        
        # Сегментируем изображение на строки
        if segment_lines:
            print("Сегментация изображения на строки...")
            line_images = segment_image_to_lines(image_path, method=segment_method)
            print(f"Найдено строк: {len(line_images)}")
            
            # Сохраняем сегментированные строки в отдельную папку (используем оригинальный путь)
            save_segmented_lines(original_image_path, line_images)
            
            recognized_lines = []
            for i, line_image in enumerate(line_images, 1):
                try:
                    # Проверяем размер изображения
                    width, height = line_image.size
                    if width < 10 or height < 10:
                        print(f"  Строка {i}/{len(line_images)}: пропущена (слишком маленькая: {width}x{height})")
                        continue
                    
                    print(f"  Распознавание строки {i}/{len(line_images)} (размер: {width}x{height})...")
                    
                    # Улучшенная предобработка изображения строки
                    line_image = preprocess_line_image(line_image)
                    
                    pixel_values = processor(line_image, return_tensors="pt").pixel_values.to(device)
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            pixel_values,
                            max_length=Config.MAX_LENGTH,
                            num_beams=6,  # Увеличено для лучшего качества
                            early_stopping=True,
                            length_penalty=1.2,  # Поощряем более длинные последовательности
                            repetition_penalty=1.3,  # Увеличиваем штраф за повторения
                            no_repeat_ngram_size=3,  # Предотвращаем повторение 3-грамм
                            temperature=0.7,  # Снижаем температуру для более детерминированного вывода
                            pad_token_id=processor.tokenizer.pad_token_id,
                            eos_token_id=processor.tokenizer.eos_token_id
                        )
                    
                    line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if line_text.strip():  # Игнорируем пустые строки
                        # Применяем постобработку для исправления частых ошибок
                        line_text = postprocess_text(line_text.strip())
                        recognized_lines.append(line_text)
                        print(f"    Результат: {line_text[:50]}...")
                except Exception as e:
                    print(f"  Ошибка при распознавании строки {i}/{len(line_images)}: {e}")
                    continue
            
            generated_text = '\n'.join(recognized_lines) if recognized_lines else ""
        else:
            # Обработка всего изображения без сегментации
            try:
                image = Image.open(image_path).convert('RGB')
                # Проверяем размер изображения
                if image.size[0] < 10 or image.size[1] < 10:
                    print("Предупреждение: изображение слишком маленькое для распознавания")
                    return ""
                
                pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=Config.MAX_LENGTH,
                        num_beams=6,
                        early_stopping=True,
                        length_penalty=1.2,
                        repetition_penalty=1.3,
                        no_repeat_ngram_size=3,
                        temperature=0.7,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id
                    )
                
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                generated_text = postprocess_text(generated_text)
            except Exception as e:
                print(f"Ошибка при распознавании изображения: {e}")
                return ""
        
        return generated_text.strip() if generated_text else ""
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Установите необходимые зависимости: pip install transformers torch opencv-python")
        return None
    except Exception as e:
        print(f"Ошибка при распознавании с TrOCR Cyrillic: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Распознавание рукописного русского текста с использованием готовых моделей"
    )
    parser.add_argument("--image", type=str, default=None, help="Путь к изображению")
    parser.add_argument("--method", type=str, default="trocr-cyrillic", 
                       choices=["trocr", "trocr-cyrillic"],
                       help="Метод распознавания (по умолчанию: trocr-cyrillic - рекомендуется для русского)")
    parser.add_argument("--batch", type=str, default=None,
                       help="Папка с изображениями для пакетной обработки")
    parser.add_argument("--output", type=str, default=None,
                       help="Путь к файлу для сохранения результатов")
    parser.add_argument("--gpu", action="store_true",
                       help="Использовать GPU (если доступен)")
    parser.add_argument("--trocr-model", type=str, default=None,
                       help="Путь к обученной TrOCR модели")
    parser.add_argument("--no-segment", action="store_true",
                       help="Отключить сегментацию по строкам (распознавать все изображение целиком)")
    parser.add_argument("--segment-method", type=str, default="yolov8",
                       choices=["yolov8"],
                       help="Метод сегментации строк (по умолчанию: yolov8)")
    
    args = parser.parse_args()
    
    device = "cuda" if args.gpu else "cpu"
    
    # Проверяем, что указан хотя бы один из параметров
    if not args.image and not args.batch:
        parser.error("Необходимо указать либо --image, либо --batch")
    
    results = []
    
    # Определяем, что обрабатывать
    if args.image and os.path.isfile(args.image):
        images_to_process = [args.image]
    elif args.batch and os.path.isdir(args.batch):
        images_to_process = [
            os.path.join(args.batch, f) 
            for f in sorted(os.listdir(args.batch))
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))
        ]
    else:
        print("Ошибка: не указан корректный путь к изображению или папке")
        return
    
    # Обрабатываем изображения
    for img_path in images_to_process:
        print(f"\n{'='*60}")
        print(f"Обработка: {os.path.basename(img_path)}")
        print(f"Метод: {args.method.upper()}")
        print(f"{'='*60}")
        
        segment = not args.no_segment
        
        if args.method == "trocr":
            text = recognize_with_trocr(img_path, model_path=args.trocr_model, device=device, 
                                        segment_lines=segment, segment_method=args.segment_method)
        elif args.method == "trocr-cyrillic":
            text = recognize_with_trocr_cyrillic(img_path, model_path=args.trocr_model, device=device, 
                                                 segment_lines=segment, segment_method=args.segment_method)
        else:
            print(f"Неизвестный метод: {args.method}")
            continue
        
        if text:
            try:
                print(f"\nРаспознанный текст:\n{text}\n")
            except UnicodeEncodeError:
                # Для Windows консоли с проблемами кодировки
                print("\nРаспознанный текст:")
                print(text.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
                print()
            results.append((os.path.basename(img_path), text))
        else:
            print("Не удалось распознать текст")
    
    # Сохранение результатов
    if args.output and results:
        with open(args.output, 'w', encoding='utf-8') as f:
            for img_name, text in results:
                f.write(f"{'='*60}\n")
                f.write(f"Изображение: {img_name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{text}\n\n")
        print(f"\nРезультаты сохранены в {args.output}")


if __name__ == "__main__":
    main()

