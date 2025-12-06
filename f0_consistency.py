import os
import json
import librosa
import numpy as np
from scipy.stats import pearsonr

# Define base paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Script location
INPUT_AUDIO_PATH = os.path.join(BASE_DIR, "input_audio.wav")  # Input audio file
OUTPUT_AUDIO_PATH = os.path.join(BASE_DIR, "output_audio.wav")  # Output audio file
EVAL_DIR = os.path.join(BASE_DIR, "data", "eval")  # Evaluation results directory

# Ensure the output directory exists
os.makedirs(EVAL_DIR, exist_ok=True)

# Function to find a unique filename (prevent overwriting existing files)
def get_unique_path(base_path, extension):
    """
    Generate a unique file path by appending numbers (e.g., _1, _2) if the file already exists.
    """
    counter = 1
    unique_path = f"{base_path}{extension}"
    while os.path.exists(unique_path):
        unique_path = f"{base_path}_{counter}{extension}"
        counter += 1
    return unique_path

# Get unique filenames for JSON and TXT results
JSON_OUTPUT = get_unique_path(os.path.join(EVAL_DIR, "f0_consistency"), ".json")
TXT_OUTPUT = get_unique_path(os.path.join(EVAL_DIR, "f0_consistency"), ".txt")

# Function to extract f0 using librosa.pyin()
def extract_f0(audio_path):
    """
    Extract f0 (pitch) values from an audio file using librosa.pyin().
    Returns an array of f0 values.
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000)  # Load audio at 16kHz
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0 = np.nan_to_num(f0)  # Replace NaNs with 0
        return f0
    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {str(e)}")
        return None

# Compute Pearson correlation between original and converted f0 values
def compute_f0_consistency(original_f0, converted_f0):
    """
    Compute Pearson correlation coefficient between original and converted f0.
    """
    if original_f0 is None or converted_f0 is None or len(original_f0) == 0 or len(converted_f0) == 0:
        return None
    min_length = min(len(original_f0), len(converted_f0))
    return pearsonr(original_f0[:min_length], converted_f0[:min_length])[0]  # Pearson correlation

if __name__ == "__main__":
    print("🔍 Computing f0 consistency using Pearson correlation coefficient...")

    # Extract f0 from input and output audio
    input_f0 = extract_f0(INPUT_AUDIO_PATH)
    output_f0 = extract_f0(OUTPUT_AUDIO_PATH)

    # Compute Pearson correlation
    pearson_corr = compute_f0_consistency(input_f0, output_f0)

    results = {"input_audio vs output_audio": pearson_corr}

    # Save results to TXT
    with open(TXT_OUTPUT, "w") as txt_file:
        txt_file.write("Filename\tPearson Correlation\n")
        txt_file.write(f"input_audio.wav vs output_audio.wav\t{pearson_corr:.4f}\n")

    # Save results to JSON
    with open(JSON_OUTPUT, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"\n f0 consistency measurement completed! Results saved to:\n {JSON_OUTPUT}\n {TXT_OUTPUT}")
    print(f"Pearson Correlation (f0 Consistency): {pearson_corr:.4f}")
