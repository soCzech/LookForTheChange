import torch
import _lookforthechange_ops


def optimal_state_change(state_tensor, action_tensor, lens, delta, kappa, max_action_state_distance=500):
    return _lookforthechange_ops.optimal_state_change(
        state_tensor.contiguous(), action_tensor.contiguous(), lens, delta, kappa, max_action_state_distance)
