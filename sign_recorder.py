import pandas as pd

class SignRecorder(object):
    def __init__(self, reference_signs: pd.DataFrame, seq_len=50):
        # Variables for recording
        self.is_recording = False
        self.seq_len = seq_len

        # List of results stored each frame
        self.recorded_results = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs

    def process_results(self, results) -> (str, bool):
        if len(self.recorded_results) < self.seq_len:
            self.recorded_results.append(results)
