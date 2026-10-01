import unittest
import numpy as np
from ca_cfar import ca_cfar_1d

class TestCACFAR(unittest.TestCase):
    def test_ca_cfar_detection(self):
        # Parameters
        num_guard_cells = 2
        num_training_cells = 10
        pfa = 1e-3

        # Create a mock signal with noise
        np.random.seed(42)
        n_samples = 200
        # Generate exponential noise which models square-law detected Gaussian noise
        noise_power = 1.0
        signal = np.random.exponential(scale=noise_power, size=n_samples)

        # Inject targets
        target_indices = [50, 120, 160]
        target_snr_db = 20 # 20 dB SNR
        target_power = noise_power * (10 ** (target_snr_db / 10))

        for idx in target_indices:
            signal[idx] += target_power

        # Run CA-CFAR
        detections, thresholds = ca_cfar_1d(signal, num_guard_cells, num_training_cells, pfa)

        # Check that targets are detected
        for idx in target_indices:
            self.assertTrue(detections[idx], f"Target at index {idx} not detected.")

        # Check for false alarms (ideally should be very few or none in this short signal)
        detected_indices = np.where(detections)[0]
        false_alarms = set(detected_indices) - set(target_indices)
        print(f"\nDetected targets at indices: {detected_indices}")
        print(f"False alarms at indices: {false_alarms}")
        self.assertLess(len(false_alarms), 5, "Too many false alarms.")

    def test_no_targets(self):
        # Parameters
        num_guard_cells = 2
        num_training_cells = 10
        pfa = 1e-4

        # Create a pure noise signal
        np.random.seed(42)
        n_samples = 200
        noise_power = 1.0
        signal = np.random.exponential(scale=noise_power, size=n_samples)

        # Run CA-CFAR
        detections, thresholds = ca_cfar_1d(signal, num_guard_cells, num_training_cells, pfa)

        detected_indices = np.where(detections)[0]
        print(f"\n[No targets test] Detected targets at indices: {detected_indices}")
        # At PFA = 1e-4, we expect roughly 0 false alarms for 200 samples
        self.assertEqual(len(detected_indices), 0, "Expected no targets to be detected.")

if __name__ == '__main__':
    unittest.main()
