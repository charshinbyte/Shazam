from processor import savehash, song2hash
from mongodb_client import collection
import utils as utils
from config import get_config
import asyncio

async def savehash(hashed_song):
    for hash_key, metadata_list in hashed_song.items():
        doc_id = str(hash_key)
        result = await collection.update_one(
            {"_id": doc_id},
            {"$push": {"metadata": {"$each": metadata_list}}},
            upsert=True
        )

async def upload_song(audio, artist, songname, config):
    utils.convert_mp3_to_wav(audio, "Upload.wav")
    song_hash = song2hash("Upload.wav", artist, songname, config)
    savehash(song_hash)


if __name__ == "__main__":
    config = get_config()

    hashed_song = song2hash("Upload.wav", "random", "random", config)
    asyncio.run(savehash(hashed_song))