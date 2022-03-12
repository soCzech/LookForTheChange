import torch
import torch.nn.functional as F
import lookforthechange


class MultiClassMLP(torch.nn.Module):

    def __init__(self, layers, n_classes=1):
        super(MultiClassMLP, self).__init__()

        self.state = torch.nn.ModuleDict({
            f"l{i:d}": torch.nn.Linear(layers[i], layers[i+1], bias=True)
            for i in range(len(layers) - 1)
        })
        self.action = torch.nn.ModuleDict({
            f"l{i:d}": torch.nn.Linear(layers[i], layers[i+1], bias=True)
            for i in range(len(layers) - 1)
        })

        self.state_layer = torch.nn.Linear(layers[-1], 2 * n_classes, bias=True)
        self.action_layer = torch.nn.Linear(layers[-1], 1 * n_classes, bias=True)

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.state)):
            x = self.state[f"l{i:d}"](x)
            x = torch.relu(x)
        state = self.state_layer(x)

        x = inputs
        for i in range(len(self.action)):
            x = self.action[f"l{i:d}"](x)
            x = torch.relu(x)
        action = self.action_layer(x)

        return {"state": state, "action": action}


class LookForTheChangeLoss(torch.nn.Module):

    def __init__(self, delta, kappa, action_loss_weight=0.2, action_class_weight=10.):
        super(LookForTheChangeLoss, self).__init__()

        self.delta = delta
        self.kappa = kappa
        self.action_loss_weight = action_loss_weight
        self.action_class_weight = action_class_weight

    def forward(self, s1s2_log_probs, action_log_probs, lens, video_level_scores=None):

        s1s2_probs = torch.softmax(s1s2_log_probs, -1)
        action_probs = torch.sigmoid(action_log_probs)

        state_targets, action_targets = lookforthechange.optimal_state_change(
            s1s2_probs, action_probs, lens, delta=self.delta, kappa=self.kappa)

        state_targets = state_targets.long()
        action_targets = action_targets.long().unsqueeze(2)

        state_loss = F.cross_entropy(
            s1s2_log_probs.view(-1, s1s2_log_probs.shape[-1]), (state_targets - 1).clamp(min=0).view(-1), reduction="none")
        state_loss = state_loss.view(state_targets.shape) * (state_targets != 0).float()

        action_loss = F.binary_cross_entropy_with_logits(
            action_log_probs, (action_targets == 2).float(), reduction="none")
        action_loss = action_loss * ((action_targets == 1).float() + self.action_class_weight * (action_targets == 2).float())

        if video_level_scores is not None:
            bs, max_len, _ = action_targets.shape  # _ == 1
            state_loss *= video_level_scores.view(bs, 1)
            action_loss *= video_level_scores.view(bs, 1, 1)

        state_loss = torch.sum(state_loss)
        action_loss = self.action_loss_weight * torch.sum(action_loss)

        return state_loss, action_loss
