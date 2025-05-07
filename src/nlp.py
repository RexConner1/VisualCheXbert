import torch
from transformers import BertTokenizer, BertForSequenceClassification

class ReportBERT:
    def __init__(self, model_name='dmis-lab/biobert-base-cased-v1.1', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=14,
            problem_type="multi_label_classification"
        ).to(self.device)

    def train(self, train_loader, val_loader, optimizer, scheduler, epochs=3):
        criterion = torch.nn.BCEWithLogitsLoss()
        best_val_macro = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                inputs = self.tokenizer(
                    batch['text'],
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(**inputs).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                total_loss += loss.item() * labels.size(0)

            avg_train_loss = total_loss / len(train_loader.dataset)
            val_metrics = self.evaluate(val_loader)
            if val_metrics['macro_f1'] > best_val_macro:
                best_val_macro = val_metrics['macro_f1']
                torch.save(self.model.state_dict(), 'models/bert_checkpoint/best.pt')

            print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_macro_f1={val_metrics['macro_f1']:.4f}")

    def predict_logits(self, texts, batch_size=16):
        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i+batch_size]
                inputs = self.tokenizer(chunk, padding=True, truncation=True, return_tensors='pt').to(self.device)
                logits = self.model(**inputs).logits
                all_logits.append(logits.cpu().numpy())
        return np.vstack(all_logits)

    def evaluate(self, data_loader):
        from src.evaluate import compute_metrics
        y_true, y_pred_probs = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                inputs = self.tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt').to(self.device)
                logits = self.model(**inputs).logits.cpu().numpy()
                y_pred_probs.append(logits)
                y_true.append(batch['labels'].cpu().numpy())
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred_probs)
        f1_per, macro_f1, _, _ = compute_metrics(y_true, (y_pred > 0.5).astype(int))
        return {'f1_per': f1_per, 'macro_f1': macro_f1}
