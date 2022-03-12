import math
import torch
import numpy as np


def constrained_argmax(pred_action, pred_state, also_only_state=False):
    max_val_, best_idx_ = -1, (0, 0, 0)
    for i in range(len(pred_state)):
        for j in range(i + 2, len(pred_state)):
            val_ = pred_state[i, 0] * pred_state[j, 1]
            k = np.argmax(pred_action[i + 1:j])
            val_ *= pred_action[i + 1 + k]
            if val_ > max_val_:
                best_idx_ = i, j, i + 1 + k
                max_val_ = val_
    if not also_only_state:
        return best_idx_

    max_val_state_, best_idx_state_ = -1, (0, 0)
    for i in range(len(pred_state)):
        for j in range(i + 1, len(pred_state)):
            val_ = pred_state[i, 0] * pred_state[j, 1]
            if val_ > max_val_state_:
                best_idx_state_ = i, j
                max_val_state_ = val_
    return best_idx_, best_idx_state_


class JointMeter:
    def __init__(self, n_classes):
        self._dict = {
            "sp": [[] for _ in range(n_classes)],
            "ap": [[] for _ in range(n_classes)],
            "jsp": [[] for _ in range(n_classes)],
            "jap": [[] for _ in range(n_classes)],
            "acc": [[] for _ in range(n_classes)]
        }

    def log(self, pred_action, pred_state, annotations, category):
        assert len(annotations) == len(pred_action) == len(pred_state)

        # state accuracy
        pred_state_idx = np.argmax(pred_state, axis=-1)
        n_gt_states = np.logical_or(annotations == 1, annotations == 3).sum()
        state_acc = ((pred_state_idx[annotations == 1] == 0).sum() +
                     (pred_state_idx[annotations == 3] == 1).sum()) / n_gt_states if n_gt_states > 0 else 0.
        if n_gt_states > 0:
            self._dict["acc"][category].append(state_acc)

        # action precision
        self._dict["ap"][category].append(1. if annotations[np.argmax(pred_action)] == 2 else 0.)

        joint, state_only = constrained_argmax(pred_action, pred_state, also_only_state=True)

        # state precision
        self._dict["sp"][category].append((0.5 if annotations[state_only[0]] == 1 else 0.0) + \
                                          (0.5 if annotations[state_only[1]] == 3 else 0.0))
        # joint precisions
        self._dict["jap"][category].append(1. if annotations[joint[2]] == 2 else 0.)
        self._dict["jsp"][category].append((0.5 if annotations[joint[0]] == 1 else 0.0) + \
                                           (0.5 if annotations[joint[1]] == 3 else 0.0))

    def __getattr__(self, item):
        if item in self._dict:
            return np.mean(self._dict[item]) * 100
        raise NotImplementedError()

    def __getitem__(self, item):
        return [np.mean(self._dict[k][item]) * 100 for k in ["acc", "sp", "jsp", "ap", "jap"]]


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._sum += val * n
        self._count += n

    @property
    def value(self):
        if self._count == 0:
            return 0
        return self._sum / self._count


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def select_correct_classes(predictions, classes, n_classes=1):
    if n_classes == 1:
        return predictions
    # predictions: B, L, n_classes * third_dim
    B, L, dim = predictions.shape
    third_dim = dim // n_classes

    x = torch.arange(0, B).view(-1, 1, 1).repeat(1, L, third_dim)
    y = torch.arange(0, L).view(1, -1, 1).repeat(B, 1, third_dim)
    z = classes.view(-1, 1, 1).repeat(1, L, third_dim) * third_dim + torch.arange(0, third_dim).view(1, 1, third_dim)

    return predictions[x, y, z]  # B, L, third_dim
