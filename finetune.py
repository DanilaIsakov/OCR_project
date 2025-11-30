"""
Скрипт для дообучения модели TrOCR на данных из папки FineTuning

Оптимизирован для конфигурации: 4xP100 16GB NVLink, CPU E5-2660v3, RAM 64GB

Для запуска на multi-GPU используйте один из способов:

1. С torchrun (рекомендуется):
   torchrun --nproc_per_node=4 finetune.py --gpu [другие параметры]

2. С accelerate:
   accelerate config  # настроить один раз
   accelerate launch finetune.py --gpu [другие параметры]

3. На одной GPU:
   python finetune.py --gpu [другие параметры]

Параметры по умолчанию оптимизированы для 4xP100:
- batch_size=12 (на GPU)
- gradient_accumulation_steps=2
- learning_rate автоматически масштабируется (5e-5 * количество GPU)
- dataloader_num_workers=8 (для E5-2660v3)
- fp16 включен для экономии памяти
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator
)
from finetuning_dataset import FineTuningDataset
from config import Config
import json


def compute_metrics(eval_pred, processor):
    """Вычисление метрик для валидации"""
    import numpy as np
    from jiwer import wer, cer
    
    predictions, labels = eval_pred
    
    # Декодируем предсказания
    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
    
    # Декодируем метки (заменяем -100 на pad_token_id)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    
    # Вычисляем WER (Word Error Rate) и CER (Character Error Rate)
    wer_score = wer(decoded_labels, decoded_preds)
    cer_score = cer(decoded_labels, decoded_preds)
    
    return {
        "wer": wer_score,
        "cer": cer_score
    }


def main():
    parser = argparse.ArgumentParser(
        description="Дообучение модели TrOCR на данных из папки FineTuning"
    )
    parser.add_argument("--base-model", type=str, default=None,
                       help="Базовая модель для дообучения (по умолчанию используется из config.py)")
    parser.add_argument("--output-dir", type=str, default="fine_tuned_model",
                       help="Директория для сохранения дообученной модели")
    parser.add_argument("--batch-size", type=int, default=12,
                       help="Размер батча на одно устройство (по умолчанию: 12 для P100)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2,
                       help="Количество шагов накопления градиента (по умолчанию: 2 для multi-GPU)")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Скорость обучения (по умолчанию: 5e-5 * количество GPU)")
    parser.add_argument("--num-epochs", type=int, default=10,
                       help="Количество эпох")
    parser.add_argument("--warmup-steps", type=int, default=500,
                       help="Количество шагов для warmup (по умолчанию: 500 для multi-GPU)")
    parser.add_argument("--max-length", type=int, default=128,
                       help="Максимальная длина последовательности")
    parser.add_argument("--dataloader-num-workers", type=int, default=8,
                       help="Количество воркеров для загрузки данных (по умолчанию: 8 для E5-2660v3)")
    parser.add_argument("--save-steps", type=int, default=500,
                       help="Сохранять чекпоинт каждые N шагов")
    parser.add_argument("--eval-steps", type=int, default=500,
                       help="Выполнять валидацию каждые N шагов")
    parser.add_argument("--logging-steps", type=int, default=50,
                       help="Логировать каждые N шагов")
    parser.add_argument("--gpu", action="store_true",
                       help="Использовать GPU (если доступен)")
    parser.add_argument("--fine-tuning-dir", type=str, default="FineTuning",
                       help="Путь к папке FineTuning")
    
    args = parser.parse_args()
    
    # Определяем устройство и количество GPU
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count() if device == "cuda" else 0
    
    # Проверяем, запущен ли скрипт через torchrun/accelerate для multi-GPU
    is_distributed = os.environ.get("RANK") is not None or os.environ.get("LOCAL_RANK") is not None
    
    print(f"\n{'='*60}")
    print(f"🔧 КОНФИГУРАЦИЯ СИСТЕМЫ")
    print(f"{'='*60}")
    print(f"Используется устройство: {device}")
    
    if device == "cpu":
        print("⚠️  ВНИМАНИЕ: Обучение на CPU очень медленное! Используйте --gpu для ускорения.")
    else:
        print(f"✅ Обнаружено GPU: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Проверяем NVLink (для P100 это важно)
        if num_gpus > 1:
            if not is_distributed:
                print(f"\n⚠️  ВНИМАНИЕ: Обнаружено {num_gpus} GPU, но скрипт запущен не через torchrun/accelerate!")
                print(f"   Для использования всех GPU запустите:")
                print(f"   torchrun --nproc_per_node={num_gpus} finetune.py --gpu [параметры]")
                print(f"   или")
                print(f"   accelerate launch finetune.py --gpu [параметры]")
                print(f"   Будет использоваться только GPU 0")
                num_gpus = 1  # Используем только один GPU
            else:
                print(f"\n📡 Multi-GPU режим активирован")
                print(f"   Trainer автоматически использует DistributedDataParallel")
                if torch.cuda.is_available():
                    # Проверяем, есть ли NVLink (косвенно через доступность peer access)
                    try:
                        can_access = torch.cuda.can_device_access_peer(0, 1) if num_gpus > 1 else False
                        if can_access:
                            print(f"   ✅ Peer-to-peer доступ между GPU доступен (NVLink)")
                        else:
                            print(f"   ⚠️  Peer-to-peer доступ недоступен (возможно, нет NVLink)")
                    except:
                        print(f"   ℹ️  NVLink статус: проверка недоступна")
        
        print(f"{'='*60}\n")
    
    # Пути к данным
    fine_tuning_dir = args.fine_tuning_dir
    train_dir = os.path.join(fine_tuning_dir, "train")
    test_dir = os.path.join(fine_tuning_dir, "test")
    train_tsv = os.path.join(fine_tuning_dir, "train.tsv")
    test_tsv = os.path.join(fine_tuning_dir, "test.tsv")
    
    # Проверяем наличие данных
    if not os.path.exists(train_dir):
        print(f"Ошибка: папка {train_dir} не найдена")
        return
    if not os.path.exists(test_dir):
        print(f"Ошибка: папка {test_dir} не найдена")
        return
    if not os.path.exists(train_tsv):
        print(f"Ошибка: файл {train_tsv} не найден")
        return
    if not os.path.exists(test_tsv):
        print(f"Ошибка: файл {test_tsv} не найден")
        return
    
    # Определяем базовую модель
    if args.base_model:
        model_name = args.base_model
    else:
        model_name = Config.MODEL_NAME
    
    print(f"Загрузка базовой модели: {model_name}")
    
    # Загружаем процессор и модель
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Настраиваем специальные токены
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Перемещаем модель на устройство
    # Примечание: при использовании Trainer с multi-GPU, модель будет автоматически
    # распределена через DistributedDataParallel, поэтому не нужно вручную перемещать
    if num_gpus == 0:
        model = model.to(device)
    
    print("Создание датасетов...")
    # Создаем датасеты
    train_dataset = FineTuningDataset(
        images_dir=train_dir,
        tsv_file=train_tsv,
        processor=processor,
        max_length=args.max_length
    )
    
    test_dataset = FineTuningDataset(
        images_dir=test_dir,
        tsv_file=test_tsv,
        processor=processor,
        max_length=args.max_length
    )
    
    print(f"Размер обучающего датасета: {len(train_dataset)}")
    print(f"Размер тестового датасета: {len(test_dataset)}")
    
    # Вычисляем learning rate с учетом количества GPU (linear scaling rule)
    if args.learning_rate is None:
        base_lr = 5e-5
        effective_num_gpus = max(num_gpus, 1)  # Минимум 1 для расчета
        learning_rate = base_lr * effective_num_gpus
    else:
        learning_rate = args.learning_rate
    
    # Вычисляем общее количество шагов с учетом multi-GPU
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * max(num_gpus, 1)
    steps_per_epoch = len(train_dataset) // effective_batch_size
    total_steps = steps_per_epoch * args.num_epochs
    
    print(f"\n📊 ПАРАМЕТРЫ ОБУЧЕНИЯ:")
    print(f"{'='*60}")
    print(f"   Батч размер (на устройство): {args.batch_size}")
    print(f"   Количество GPU: {max(num_gpus, 1)}")
    print(f"   Накопление градиента: {args.gradient_accumulation_steps}")
    print(f"   Эффективный размер батча: {effective_batch_size}")
    print(f"   Learning rate: {learning_rate:.2e} {'(масштабирован для multi-GPU)' if num_gpus > 1 and args.learning_rate is None else ''}")
    print(f"   Эпох: {args.num_epochs}")
    print(f"   Шагов на эпоху: ~{steps_per_epoch}")
    print(f"   Всего шагов: ~{total_steps}")
    print(f"   Warmup steps: {args.warmup_steps}")
    print(f"   Воркеров для загрузки данных: {args.dataloader_num_workers if device == 'cuda' else 0}")
    print(f"   Mixed Precision (FP16): {'Да' if torch.cuda.is_available() else 'Нет'}")
    print(f"{'='*60}\n")
    
    # Настройки обучения, оптимизированные для 4xP100 16GB NVLink
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",
        # FP16 для экономии памяти на P100 (поддерживается)
        fp16=torch.cuda.is_available(),
        # Оптимизации для multi-GPU с NVLink
        dataloader_num_workers=args.dataloader_num_workers if device == "cuda" else 0,
        dataloader_pin_memory=torch.cuda.is_available(),  # Ускоряет передачу данных на GPU
        # Для multi-GPU: автоматически использует DistributedDataParallel
        ddp_find_unused_parameters=False,  # Ускоряет обучение на multi-GPU
        # Оптимизации производительности
        remove_unused_columns=False,  # Важно для seq2seq моделей
        # Для больших моделей и multi-GPU
        gradient_checkpointing=False,  # Можно включить, если не хватает памяти
        # Сохранение и логирование
        save_total_limit=3,  # Хранить только последние 3 чекпоинта
        logging_first_step=True,
        # Оптимизация для NVLink
        dataloader_drop_last=True,  # Избегаем проблем с синхронизацией на последнем батче
    )
    
    # Создаем функцию для вычисления метрик с процессором
    def compute_metrics_with_processor(eval_pred):
        return compute_metrics(eval_pred, processor)
    
    # Создаем тренер
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_with_processor,
    )
    
    print("🚀 Начало обучения...")
    if num_gpus > 1:
        print(f"   Используется распределенное обучение на {num_gpus} GPU")
        print(f"   Эффективный размер батча: {effective_batch_size}")
    print()
    
    # Обучаем модель
    train_result = trainer.train()
    
    # Сохраняем финальную модель
    print(f"Сохранение модели в {args.output_dir}...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    # Сохраняем метрики
    metrics = train_result.metrics
    metrics_file = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Обучение завершено!")
    print(f"Модель сохранена в: {args.output_dir}")
    print(f"Метрики сохранены в: {metrics_file}")
    
    # Выполняем финальную оценку
    print("\nВыполнение финальной оценки на тестовом датасете...")
    eval_metrics = trainer.evaluate()
    print(f"Финальные метрики:")
    print(f"  WER (Word Error Rate): {eval_metrics.get('eval_wer', 'N/A'):.4f}")
    print(f"  CER (Character Error Rate): {eval_metrics.get('eval_cer', 'N/A'):.4f}")


if __name__ == "__main__":
    main()

