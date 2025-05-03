import os
import pdfplumber
import docx
import whisper
from pydub import AudioSegment
import tempfile
from werkzeug.utils import secure_filename

whisper_model = whisper.load_model("base")

ALLOWED_DOC_EXTENSIONS = {'.pdf', '.docx'}
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.ogg'}

def is_allowed_file(filename, allowed_extensions):
    return os.path.splitext(filename)[1].lower() in allowed_extensions

def read_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return read_pdf(file_path)
    elif ext == '.docx':
        return read_docx(file_path)
    else:
        return "Unsupported document format."

def transcribe_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio.export(tmp.name, format="wav")
            result = whisper_model.transcribe(tmp.name)
        os.remove(tmp.name)
        return result["text"]
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def handle_file_upload(file, upload_dir="uploads"):
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_dir, filename)
    os.makedirs(upload_dir, exist_ok=True)
    file.save(file_path)
    return file_path
