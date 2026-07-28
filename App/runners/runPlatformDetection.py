"""Runner for PlatformDetection block - detects mzNPS vs mechanoNPS platform."""

import numpy as np


def run(inputs, params, block):
    data = inputs.get("data")
    sample_rate = inputs.get("sampleRate", 1)
    amps_per_volt = inputs.get("ampsPerVolt", 1e-7)

    mz_batch_size = params.get("mzBatchSize", 7)
    use_current_col = params.get("useCurrentCol", True)
    current_col_1based = params.get("currentCol", 1)

    if data is None:
        raise ValueError("No data provided to PlatformDetection.")

    # Ensure data is a numpy array
    data = np.asarray(data, dtype=float).ravel()

    if sample_rate <= 15000:
        # Multizone NPS (typically 10 kHz)
        platform = "mzNPS"

        # Reshape: each row is one batch of mzBatchSize samples
        n_batches = len(data) // mz_batch_size
        data_reshaped = data[:n_batches * mz_batch_size].reshape(
            n_batches, mz_batch_size)

        if use_current_col:
            # Convert 1-based column index to 0-based
            current_col = int(current_col_1based) - 1
            if current_col < 0 or current_col >= mz_batch_size:
                raise ValueError(
                    f"currentCol must be between 1 and {mz_batch_size}, "
                    f"got {current_col_1based}.")

            current = data_reshaped[:, current_col] * amps_per_volt
            voltage_cols = [i for i in range(mz_batch_size)
                           if i != current_col]
            voltage = data_reshaped[:, voltage_cols]
            with np.errstate(divide="ignore", invalid="ignore"):
                resistance = voltage / current[:, np.newaxis]   # R = V / I
            resistance = np.nan_to_num(resistance, nan=0.0,
                                       posinf=0.0, neginf=0.0)
            data_out = resistance
        else:
            # No current column — each column is a zone
            data_out = data_reshaped
    else:
        # Mechano NPS (typically 50 kHz)
        platform = "mechanoNPS"
        data_out = data

    return {"data": data_out, "platform": platform, "sampleRate": sample_rate}
