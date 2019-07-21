import torch
import numpy as np

def jaccard_index(target, pred, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, num_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = pred_inds[target_inds].long().sum().data.cpu()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu() + target_inds.long().sum().data.cpu() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(union))
    return np.array(ious)

def rmse(target, pred):
    return torch.sqrt(torch.sum((target - pred)**2))

def angle_error(target, pred):
    target = target.data.cpu().numpy()
    pred = pred.data.cpu().numpy()
    angle = np.arccos(np.dot(target, pred) / (np.linalg.norm(target) * np.linalg.norm(pred)))
    return angle
