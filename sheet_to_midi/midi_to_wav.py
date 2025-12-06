import os
import subprocess

def midi_to_wav(midi_path, wav_path, soundfont_path="resources/soundfonts/FluidR3_GM.sf2"):
    """Convert MIDI to WAV using fluidsynth and a SoundFont (.sf2) file."""
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    if not os.path.exists(soundfont_path):
        raise FileNotFoundError(f"SoundFont file not found: {soundfont_path}")

    command = [
        "fluidsynth",
        "-ni", soundfont_path,
        midi_path,
        "-F", wav_path,
        "-r", "44100"
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("[ERROR] fluidsynth output:")
        print(result.stderr.decode())
        raise RuntimeError("fluidsynth conversion failed")
    else:
        print(f"[INFO] Successfully converted {midi_path} to {wav_path}")


# Example usage (during debug or Flask integration):
if __name__ == "__main__":
    midi_to_wav(
        midi_path="static/sample.mid",
        wav_path="static/sample.wav"
    )
