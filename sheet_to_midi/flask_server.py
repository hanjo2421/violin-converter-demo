from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid

from midi_converter import convert 
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

    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    output_prefix = os.path.splitext(filename)[0]
    midi_filename = f"{output_prefix}.mid"
    midi_path = os.path.join(STATIC_FOLDER, midi_filename)
    wav_filename = f"{output_prefix}.wav"
    wav_path = os.path.join(STATIC_FOLDER, wav_filename)
    debug_image_name = f"{output_prefix}_debug_notes.png"
    debug_image_path = os.path.join(STATIC_FOLDER, debug_image_name)

    success = convert(
        img_path=filepath,
        out_mid=midi_path,
        debug=True,
        force_all=False
    )

    if not success:
        return jsonify({'error': 'Conversion failed'}), 500

    midi_to_wav(midi_path, wav_path)

    if not os.path.exists(debug_image_path):
        debug_image_name = None

    return jsonify({
        'midiFile': f"/static/{midi_filename}",
        'audioFile': f"/static/{wav_filename}",
        'debugImage': f"/static/{debug_image_name}" if debug_image_name else None
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=5050, debug=True)
