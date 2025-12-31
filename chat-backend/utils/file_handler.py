from __future__ import annotations
import uuid
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from typing import List, Dict

ALLOWED_EXTENSIONS = {'pdf', 'json'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

class FileHandler:
    def __init__(self, upload_dir: Path):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def allowed_file(self, filename: str) -> bool:
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def generate_session_id(self) -> str:
        return str(uuid.uuid4())
    
    def save_uploaded_files(self, files) -> Dict:
        """
        Save uploaded files and return session info
        """
        session_id = self.generate_session_id()
        session_dir = self.upload_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        errors = []
        
        for file in files:
            if file and self.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = session_dir / filename
                
                try:
                    file.save(str(filepath))
                    file_size = filepath.stat().st_size
                    
                    if file_size > MAX_FILE_SIZE:
                        filepath.unlink()
                        errors.append(f"File too large: {filename}")
                        continue
                    
                    file_type = filepath.suffix[1:].lower()
                    saved_files.append({
                        'filename': filename,
                        'path': str(filepath),
                        'type': file_type,
                        'size': file_size
                    })
                except Exception as e:
                    errors.append(f"Failed to save {filename}: {str(e)}")
            else:
                errors.append(f"Invalid file type: {file.filename if file else 'unknown'}")
        
        return {
            'session_id': session_id,
            'session_dir': str(session_dir),
            'files': saved_files,
            'errors': errors if errors else None
        }
    
    def get_session_files(self, session_id: str) -> List[Dict]:
        """Get all files in a session directory"""
        session_dir = self.upload_dir / session_id
        if not session_dir.exists():
            return []
        
        files = []
        for file_path in session_dir.iterdir():
            if file_path.is_file():
                files.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'type': file_path.suffix[1:].lower(),
                    'size': file_path.stat().st_size
                })
        return files
    
    def delete_session(self, session_id: str) -> bool:
        """Delete all files in a session"""
        session_dir = self.upload_dir / session_id
        if not session_dir.exists():
            return False
        
        try:
            for file_path in session_dir.iterdir():
                file_path.unlink()
            session_dir.rmdir()
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
