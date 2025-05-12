from fastapi import FastAPI, UploadFile, File, Form
from config import get_config
from processor import match_clip, upload_song, save_song_to_sqlite, delete_songs
from mangum import Mangum
import hashlib
from datetime import datetime
from fastapi.responses import HTMLResponse
import pathlib
import os
import utils

def generate_song_id(artist: str, song_name: str, timestamp: str = None) -> str:
    if not timestamp:
        timestamp = datetime.now().isoformat()
    base_str = f"{artist.lower().strip()}::{song_name.lower().strip()}::{timestamp}"
    return hashlib.md5(base_str.encode()).hexdigest()

app = FastAPI()
config = get_config()

@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    html_path = pathlib.Path("static/index.html")
    return HTMLResponse(content=html_path.read_text(), status_code=200)

@app.get("/")
async def home():
    return {"message": "API Successfully turned on"}

@app.post('/record')
async def record_endpoint(file: UploadFile = File(...)):
    with open("record.webm", "wb") as f:
        content = await file.read()
        f.write(content)

    utils.convert_webm_to_wav("record.webm", "record.wav")

    matched_song = await match_clip("record.wav", config)


    return {"Matched Song " : matched_song}

@app.post("/match")
async def match_endpoint(file: UploadFile = File(...)):    
    with open("Snippet.wav", "wb") as f:
        content = await file.read()
        f.write(content)
    
    matched_song = await match_clip("Snippet.wav", config)
    os.remove("Snippet.wav")

    return {"Matched Song " : matched_song}

@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...), artist: str = Form(...), song_name: str = Form(...)):
    """
        Adds a song to the MongoDB database
    """  
    try:
        with open("Upload.mp3", "wb") as f:
            content = await file.read()
            f.write(content)

        song_id = generate_song_id(artist, song_name)
        save_song_to_sqlite("data/songs.db", song_id, artist, song_name)
        
        await upload_song("Upload.mp3", song_id, config)

        os.remove("Upload.mp3")
        os.remove("Upload.wav")

        return {"message" : "Song Uploaded Successfully"}
    except Exception as e:
        return {"error": f"Failed to upload song: {str(e)}"}

@app.delete("/delete")
async def delete_entries():
    await delete_songs()
    return {"message" : "all songs deleted"}
