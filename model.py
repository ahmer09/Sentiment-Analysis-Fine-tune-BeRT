import os
import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, BertTokenizer

PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index

LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
GRADIENT_ACCUMALATION_STEPS = 3
MAX_GRAD_NORM = 1.0
SEED = 42
DEVICE = "gpu"

bert_dir = 'C://Users//Hammer//PycharmProjects//Bert-fine-tune//model//cased_L-12_H-768_A-12//cased_L-12_H-768_A-12//'


class FinBERT:
    model = None
    tokenizer = None

    def __init__(self):
        self.pad_token_label_id = PAD_TOKEN_LABEL_ID
        self.device = torch.device(DEVICE)

    def predict(self, input_ids, mapping):
        if self.model is None or self.tokenizer is None:
            self.load()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids=input_ids)

        res_class = np.argmax(logits[0].detach().cpu().numpy(), axis=1)
        res_proba = torch.sigmoid(logits[0].unsqueeze(-1)).detach().cpu().numpy()[0]
        res_proba = np.round(res_proba, 4)
        sentiment_score = res_proba[1] - res_proba[0]
        print('---Class "{}" with probality {:.2f}.'.format(mapping.get(res_class[0]), np.max(res_proba)))
        print(' ---Probablity score: {}'.format(res_proba.tolist()))
        return {"logits": logits[0], "prob_predict": res_proba, "sentiment_score": sentiment_score, "predict": mapping.get(res_class[0])}

    def evaluate(self, dataloader, y_true, label_names):
        _, y_pred = self._predict_tags_batched(dataloader)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print({"acc":acc, "f1": round(f1, 4)})
        print(classification_report(y_true, y_pred, target_names=label_names))
        return

    def _predict_tags_batched(self, dataloader):
        preds = None
        for batch in tqdm(dataloader, desc="Evaluating: "):
            with torch.no_grad():
                outputs = self.model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], labels=batch[3])
                _, logits = outputs[:2]
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        preds_final = np.append(preds, axis=1)
        return logits, preds_final


    def train(self, tokenizer, dataloader, model, epochs):
        assert self.model is None
        model.to(self.device)
        self.model = model
        self.tokenizer = tokenizer

        t_total = len(dataloader)// GRADIENT_ACCUMALATION_STEPS * epochs

        # Prepare optimizer and schedule
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=t_total)

        model.train()
        print("Training..")
        for _e in range(epochs):
            print('====== Epoch {:}/{:} ======='.format(_e + 1, epochs))
            for step, batch in enumerate(tqdm(dataloader, desc='Iteration')):
                outputs = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], labels=batch[3])
                loss = outputs[0]

                if GRADIENT_ACCUMALATION_STEPS > 1:
                    loss = loss/GRADIENT_ACCUMALATION_STEPS
                loss.backward()

                if step%40 == 0 and not step == 0:
                    print('Loss: %f --- Batch {:>5,} of {:>5,}. '.format(loss, step, len(dataloader)))

                if (step + 1)%GRADIENT_ACCUMALATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
        self.model = model
        return

    def load(self, model_dir=bert_dir):
        if not os.path.exists(model_dir):
            raise FileNotFoundError("folder `{}` does not exist.".format(model_dir))

        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.to(self.device)






