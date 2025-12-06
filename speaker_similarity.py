import os
import json
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm
from datetime import datetime
from resemblyzer import VoiceEncoder, preprocess_wav

# Load the Speaker Encoder model
encoder = VoiceEncoder()

# Function to generate a unique file path (prevents overwriting)
def get_unique_path(base_path, ext):
    """
    Generates a unique file path by appending a counter to the filename if the file already exists.
    """
    counter = 1
    unique_path = f"{base_path}{ext}"
    
    while os.path.exists(unique_path):
        unique_path = f"{base_path}_{counter}{ext}"
        counter += 1
    
    return unique_path

# Function to extract speaker embeddings
def extract_embedding(audio_path):
    """
    Extracts speaker embeddings using Resemblyzer for a given audio file.
    """
    try:
        wav = preprocess_wav(audio_path)  # Preprocess the audio file
        embedding = encoder.embed_utterance(wav)  # Extract speaker embedding
        return embedding.astype(float)  # Convert to standard Python float for JSON serialization
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

# Function to compute speaker similarity
def compute_similarity(embedding1, embedding2):
    """
    Computes cosine similarity between two speaker embeddings.
    """
    return float(1 - cosine(embedding1, embedding2))  # Ensure result is a Python float

if __name__ == "__main__":
    # Define directories
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    AUDIO_ROOT = os.path.join(BASE_DIR, "data", "VCTK_2000")  # Directory containing audio files
    EVAL_DIR = os.path.join(BASE_DIR, "data", "eval")  # Directory for saving results

    # Generate unique JSON & TXT file paths
    json_base = os.path.join(EVAL_DIR, "speaker_similarity")
    txt_base = os.path.join(EVAL_DIR, "speaker_similarity")
    JSON_OUTPUT = get_unique_path(json_base, ".json")
    TXT_OUTPUT = get_unique_path(txt_base, ".txt")

    # Ensure the evaluation directory exists
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Check if the audio directory exists
    if not os.path.exists(AUDIO_ROOT):
        print(f"Error: Audio file directory not found: {AUDIO_ROOT}")
        exit(1)

    # Retrieve the list of audio files
    audio_files = [f for f in os.listdir(AUDIO_ROOT) if f.endswith(".flac") or f.endswith(".wav")]
    if len(audio_files) == 0:
        print("No audio files found. Exiting...")
        exit(1)

    # Extract speaker embeddings
    embeddings = {}
    print(f"Extracting speaker embeddings for {len(audio_files)} files...")

    try:
        for audio_file in tqdm(audio_files, desc="Extracting Embeddings", unit="file"):
            file_path = os.path.join(AUDIO_ROOT, audio_file)
            embedding = extract_embedding(file_path)
            if embedding is not None:
                embeddings[audio_file] = embedding.tolist()  # Convert NumPy array to Python list
    except KeyboardInterrupt:
        print("\n KeyboardInterrupt detected. Saving progress before exiting...")
        with open(JSON_OUTPUT, "w") as json_file:
            json.dump(embeddings, json_file, indent=4)
        print(f"Partial JSON results saved: {JSON_OUTPUT}")
        exit(1)

    # 🔍 Compute speaker similarity
    similarity_results = {}
    print(f"Computing speaker similarity for {len(audio_files)} files...")

    try:
        with open(TXT_OUTPUT, "w") as txt_file:
            txt_file.write("Speaker1\tSpeaker2\tSimilarity\n")  # Add header to TXT file

            for i in tqdm(range(len(audio_files)), desc="Computing Similarity"):
                for j in range(i + 1, len(audio_files)):
                    file1, file2 = audio_files[i], audio_files[j]
                    if file1 in embeddings and file2 in embeddings:
                        similarity = compute_similarity(np.array(embeddings[file1]), np.array(embeddings[file2]))
                        similarity_results[f"{file1} vs {file2}"] = similarity
                        
                        # Save results in TXT file
                        txt_file.write(f"{file1}\t{file2}\t{similarity:.4f}\n")

    except KeyboardInterrupt:
        print("\n KeyboardInterrupt detected. Saving progress before exiting...")
        with open(JSON_OUTPUT, "w") as json_file:
            json.dump(similarity_results, json_file, indent=4)
        print(f"Partial JSON results saved: {JSON_OUTPUT}")
        exit(1)

    # Save results in JSON file
    with open(JSON_OUTPUT, "w") as json_file:
        json.dump(similarity_results, json_file, indent=4)

    print(f"\n Speaker similarity evaluation completed! Scores saved to:\n {JSON_OUTPUT}\n {TXT_OUTPUT}")
