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

