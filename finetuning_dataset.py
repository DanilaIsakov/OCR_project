"""
Класс для загрузки и обработки датасета для дообучения из папки FineTuning
Формат: TSV файлы с парами (имя_изображения, текст)
"""
import os
import re
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor


class FineTuningDataset(Dataset):
    def __init__(self, images_dir, tsv_file, processor, max_length=512):
        """
        Args:
            images_dir: путь к папке с изображениями (train или test)
            tsv_file: путь к TSV файлу с парами (имя_изображения, текст)
            processor: TrOCRProcessor для обработки изображений и текста
            max_length: максимальная длина текстовой последовательности
        """
        self.images_dir = images_dir
        self.processor = processor
        self.max_length = max_length
        
        # Читаем TSV файл
        self.samples = []
        with open(tsv_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Разделяем по табуляции
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    image_name, text = parts
                    image_path = os.path.join(images_dir, image_name)
                    
                    # Проверяем, что файл существует
                    if os.path.exists(image_path):
                        self.samples.append((image_name, text))
                    else:
                        print(f"Предупреждение: файл {image_path} не найден, пропускаем")
                else:
                    print(f"Предупреждение: некорректная строка в TSV: {line[:50]}...")
        
        print(f"Загружено {len(self.samples)} образцов из {tsv_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Получаем имя файла и текст
        image_name, text = self.samples[idx]
        image_path = os.path.join(self.images_dir, image_name)
        
        # Загружаем изображение
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка при загрузке изображения {image_path}: {e}")
            # Возвращаем пустое изображение в случае ошибки
            image = Image.new('RGB', (100, 32), color='white')
        
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

