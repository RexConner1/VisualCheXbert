import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from src.preprocess import preprocess_chexpert
from src.vision import DenseNetTeacher
from src.nlp import ReportBERT
from src.calibration import youden_thresholds, train_logistic_calibrator, temperature_scale
from src.evaluate import compute_metrics

def main():
    # 1. Data prep (already done via preprocess.py)
    # 2. Vision inference
    teacher = DenseNetTeacher('models/densenet_chexpert.pth')
    # (pseudo-code: batch iterate over data/train.jsonl, store teacher.infer(image_path))

    # 3. NLP training
    from src.dataset import ReportDataset  # assume a simple Dataset wrapper
    train_ds = ReportDataset('data/train.jsonl', pseudo_label_dir='pseudo/')
    val_ds   = ReportDataset('data/valid.jsonl', pseudo_label_dir='pseudo/')
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16)

    student = ReportBERT()
    optimizer = AdamW(student.model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    student.train(train_loader, val_loader, optimizer, scheduler, epochs=3)

    # 4. Calibration on validation
    val_logits = student.predict_logits([rec['text'] for rec in val_ds])
    val_labels = val_ds.get_labels()  # shape (N,14)
    thresholds = youden_thresholds(val_logits, val_labels)
    calib = train_logistic_calibrator(val_logits, (val_logits > thresholds).astype(int))
    T = temperature_scale(val_logits, val_labels)

    # 5. Final evaluation on test set (omitted for brevity)
    # ...
    print("Reproduction complete.")

if __name__ == '__main__':
    main()
