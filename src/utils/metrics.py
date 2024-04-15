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
    input_batch = input_batch.cpu().numpy()
    result_batch = {k: v.cpu().numpy() for k, v in result_batch.items()}
    
    auses = []

    num_elems = input_batch[0][0].size
    perc = 1/num_elems
    y = [perc * i  for i in range(num_elems)]
    for instance_id in range(input_batch.shape[0]):
        input_instance = input_batch[instance_id][0]
        mean_result = result_batch['mean'][instance_id][0]
        var_result = result_batch['var'][instance_id][0]
        # Compute sparsification curves for the predicted depth map
        def sparsification(error, uncertainty):
            x, y = np.unravel_index(np.argsort(uncertainty, axis=None)[::-1], uncertainty.shape) # Descending order
            return np.array([error[x][y] for x, y in zip(x, y)])


        error = np.abs(input_instance - mean_result) # RMSE -> SE (without mean and root)
        
        sparsification_prediction = sparsification(error, var_result)
        sparsification_oracle = sparsification(error, error)
        
        # Normalization of the sparsification curves
        sparsification_prediction = (sparsification_prediction - np.min(sparsification_prediction)) \
                                    / (np.max(sparsification_prediction) - np.min(sparsification_prediction))
        sparsification_oracle = (sparsification_oracle - np.min(sparsification_oracle)) \
                                / (np.max(sparsification_oracle) - np.min(sparsification_oracle))
                                
        # Compute the means of the sparsification curves
        sparsification_errors_means = []
        sparsification_oracle_means = []
        sum_errors_means = np.sum(sparsification_prediction)
        sum_oracle_means = np.sum(sparsification_oracle)
        for i in range(num_elems):
            sparsification_errors_means.append(sum_errors_means / (num_elems - i))
            sparsification_oracle_means.append(sum_oracle_means / (num_elems - i))
            sum_errors_means -= sparsification_prediction[i]
            sum_oracle_means -= sparsification_oracle[i]
        # Compute the AUSE by integrating the absolute values of the error differences
        sparsification_errors = np.abs(np.array(sparsification_oracle_means) - 
                                       np.array(sparsification_errors_means))
        auses.append(np.trapz(sparsification_errors, y))

    return np.array(auses).mean()


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