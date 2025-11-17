"""
Класс для загрузки и обработки датасета с рукописными изображениями и текстами
"""
import os
import re
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor


def read_text_file(file_path):
    """
    Читает текстовый файл, пробуя разные кодировки
    
    Args:
        file_path: путь к файлу
        
    Returns:
        содержимое файла в виде строки
    """
    encodings = ['utf-8', 'windows-1251', 'cp1251', 'latin-1', 'utf-16']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            # Если это не проблема с кодировкой, пробуем следующую
            continue
    
    # Если ничего не сработало, пробуем читать как bytes и декодировать
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            # Пробуем разные кодировки
            for encoding in encodings:
                try:
                    return content.decode(encoding)
                except (UnicodeDecodeError, UnicodeError):
                    continue
    except Exception:
        pass
    
    # Последняя попытка - игнорировать ошибки
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""


class HandwritingDataset(Dataset):
    def __init__(self, images_dir, writings_dir, processor, max_length=512, train=True, train_split=0.8):
        """
        Args:
            images_dir: путь к папке с изображениями
            writings_dir: путь к папке с расшифровками
            processor: TrOCRProcessor для обработки изображений и текста
            max_length: максимальная длина текстовой последовательности
            train: флаг обучения (True) или валидации (False)
            train_split: доля данных для обучения
        """
        self.images_dir = images_dir
        self.writings_dir = writings_dir
        self.processor = processor
        self.max_length = max_length
        
        # Получаем список всех изображений
        image_files = sorted([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))])
        
        # Фильтруем только те, для которых есть расшифровка
        self.image_files = []
        for img_file in image_files:
            # Ищем соответствующий текстовый файл
            txt_file = os.path.splitext(img_file)[0] + '.txt'
            txt_path = os.path.join(writings_dir, txt_file)
            if os.path.exists(txt_path):
                self.image_files.append(img_file)
        
        # Разделяем на train/val
        split_idx = int(len(self.image_files) * train_split)
        if train:
            self.image_files = self.image_files[:split_idx]
        else:
            self.image_files = self.image_files[split_idx:]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Загружаем изображение
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        
        # Загружаем текст
        txt_file = os.path.splitext(img_file)[0] + '.txt'
        txt_path = os.path.join(self.writings_dir, txt_file)
        
        text = read_text_file(txt_path)
        
        # Очищаем текст: удаляем лишние пробелы и переносы строк
        text = text.strip()
        # Заменяем множественные пробелы на один
        text = re.sub(r'\s+', ' ', text)
        # Если текст слишком длинный, обрезаем
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        # Обрабатываем изображение и текст через processor
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Кодируем текст
        encoding = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        labels = encoding["input_ids"].squeeze()
        
        # Заменяем padding token id на -100 (игнорируется в loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

