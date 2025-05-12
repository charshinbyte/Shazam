
from .processor import save_song_to_sqlite

if __name__ == "__main__":
    song_id = "d2a873624d198caa5afc5d36dc2d519a"
    artist = "LE SSERAFIM"
    song_name = "Come Over"
    save_song_to_sqlite("data/songs.db", song_id, artist, song_name)
