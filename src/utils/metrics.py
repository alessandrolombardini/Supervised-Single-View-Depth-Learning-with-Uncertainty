import scipy
import numpy as np
import torch
from scipy.stats import t



def compute_psnr(label, result, rgb_range=1.):
    result = result['mean']
    if label.nelement() == 1: return 0
    diff = (result - label) / rgb_range
    mse = diff.pow(2).mean()
    return -10 * torch.log10(mse)


def compute_rmse(label, result):
    result = result['mean']
    if label.nelement() == 1: return 0
    mse = (result - label).pow(2).mean().sqrt()
    return mse

    
def compute_ause(input_batch, result_batch):
    """Compute the Area Under the Sparsification Error (AUSE)."""
    ause = []
    input_batch = input_batch.cpu()
    result_batch = {k: v.cpu() for k, v in result_batch.items()}
    for instance_id in range(input_batch.shape[0]):
        input_instance = input_batch[instance_id][0]
        mean_result = result_batch['mean'][instance_id][0]
        var_result = result_batch['var'][instance_id][0]
        # Compute sparsification curves for the predicted depth map
        # Sparsification
        def sparsification(error, uncertainty):
            x, y = np.unravel_index(np.argsort(uncertainty, axis=None), uncertainty.shape)
            ranking = np.stack((x, y), axis=1)
            sparsification = []
            for x, y in ranking:
                sparsification.append(error[x][y])
            return np.array(sparsification)    
        error = (input_instance - mean_result).pow(2) # RMSE -> SE (without mean and root)
        sparsification_prediction = sparsification(error,
                                                   var_result)
        sparsification_oracle = sparsification(error, 
                                               error)
        # Calculate the error difference between the sparsification curves
        sparsification_errors = sparsification_oracle - sparsification_prediction
        # Compute the AUSE by integrating the absolute values of the error differences
        ause = np.trapz(np.abs(sparsification_errors), np.arange(sparsification_errors.shape[0]))
    return ause.mean()


def compute_auce(input_batch, result_batch):
    """Compute the Area Under the Calibration Error curve (AUCE)."""
    input_batch = input_batch.detach().cpu().numpy()
    result_batch = {k: v.detach().cpu().numpy() for k, v in result_batch.items()}
    # Analyze a batch
    auces = []
    for instance_id in range(input_batch.shape[0]):
        input_instance = input_batch[instance_id][0]
        mean_pred = result_batch['mean'][instance_id][0]
        var_pred = result_batch['var'][instance_id][0]
        # Analyze an image
        counter = []
        std = np.sqrt(var_pred)
        for p in np.arange(0, 0.99, 0.01): # p è la confidenza  
            p += 0.01
            neg_bound, pos_bound = scipy.stats.norm.interval(p, mean_pred, std)
            lower_bounds = mean_pred + neg_bound
            upper_bounds = mean_pred + pos_bound
            counter.append(np.sum((input_instance <= upper_bounds) * (input_instance >= lower_bounds)))
        counter = np.array(counter) - np.array(range(0, 1, 100))
        auces.append(np.trapz(np.abs(counter), np.arange(0, 0.99, 0.01))) 
    return np.array(auces).mean()



