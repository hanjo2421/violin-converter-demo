from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os, uuid, traceback

from midi_converter_v2 import convert
from midi_to_wav import midi_to_wav

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}}, supports_credentials=True)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route('/convert', methods=['POST'])
def convert_sheet():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    ext = os.path.splitext(file.filename)[1].lower() or '.png'
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    file.save(filepath)

    output_prefix = os.path.splitext(filename)[0]
    midi_path = os.path.join(STATIC_FOLDER, f"{output_prefix}.mid")
    wav_path  = os.path.join(STATIC_FOLDER, f"{output_prefix}.wav")

    instrument = request.form.get('instrument', 'violin').lower()

    success = convert(
        img_path=filepath,
        out_mid=midi_path,
        instrument=instrument,
        debug=True,
        force_all=False
    )
    if not success:
        return jsonify({'error': 'Conversion failed'}), 500

    try:
        midi_to_wav(midi_path, wav_path)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'MIDI-to-WAV failed'}), 500

    return jsonify({
        'midiFile':  f"/static/{os.path.basename(midi_path)}",
        'audioFile': f"/static/{os.path.basename(wav_path)}",
        'debugImage': f"/static/{output_prefix}_debug_notes.png" 
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=5050, debug=True)
