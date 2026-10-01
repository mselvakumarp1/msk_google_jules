import numpy as np

def ca_cfar_1d(signal, num_guard_cells, num_training_cells, pfa):
    """
    1D Cell-Averaging Constant False Alarm Rate (CA-CFAR) detection algorithm.

    Args:
        signal (np.ndarray): 1D array of signal power (magnitude squared).
        num_guard_cells (int): Number of guard cells on ONE side of the Cell Under Test (CUT).
        num_training_cells (int): Number of training cells on ONE side of the guard cells.
        pfa (float): Desired probability of false alarm.

    Returns:
        detections (np.ndarray): Boolean array indicating detected targets (True = target).
        thresholds (np.ndarray): Threshold values computed for each cell.
    """
    signal = np.asarray(signal)
    n = len(signal)
    detections = np.zeros(n, dtype=bool)
    thresholds = np.zeros(n)

    # Total number of training cells
    n_training = 2 * num_training_cells

    # Calculate threshold factor alpha
    alpha = n_training * (pfa**(-1.0 / n_training) - 1)

    # Iterate through the signal, avoiding the edges where a full window cannot be formed
    for i in range(num_guard_cells + num_training_cells, n - (num_guard_cells + num_training_cells)):
        # Indices for the left training cells
        left_train_start = i - num_guard_cells - num_training_cells
        left_train_end = i - num_guard_cells

        # Indices for the right training cells
        right_train_start = i + num_guard_cells + 1
        right_train_end = i + num_guard_cells + num_training_cells + 1

        # Sum the noise in the training cells
        noise_sum = np.sum(signal[left_train_start:left_train_end]) + \
                    np.sum(signal[right_train_start:right_train_end])

        # Average noise power
        noise_avg = noise_sum / n_training

        # Calculate threshold
        threshold = alpha * noise_avg
        thresholds[i] = threshold

        # Detect target
        if signal[i] > threshold:
            detections[i] = True

    return detections, thresholds
