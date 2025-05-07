import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

def youden_thresholds(logits, labels):
    thresholds = []
    for i in range(labels.shape[1]):
        fpr, tpr, thr = roc_curve(labels[:,i], logits[:,i])
        youden = tpr - fpr
        idx = np.argmax(youden)
        thresholds.append(thr[idx])
    return np.array(thresholds)

def train_logistic_calibrator(logits, labels):
    # train one-vs-rest logistic regression for multi-label calibration
    calib = LogisticRegression(penalty='l2', class_weight='balanced', max_iter=500)
    calib.fit(logits, labels)
    return calib

def temperature_scale(logits, labels, lr=0.01, epochs=50):
    # single scalar temperature on logits to minimize NLL
    import torch, torch.nn.functional as F
    T = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=epochs)

    logits_t = torch.tensor(logits)
    labels_t = torch.tensor(labels)

    def _eval():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits_t / T, labels_t)
        loss.backward()
        return loss

    optimizer.step(_eval)
    return T.item()
