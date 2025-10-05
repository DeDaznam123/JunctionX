import yt_dlp
import tempfile
import os

def extract_audio_from_link(link: str):
    """
    Extracts audio from a video link using yt-dlp.
    Returns the path to the downloaded audio file, the resolved media URL, and the video title.
    """
    temp_dir = tempfile.mkdtemp()
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=True)
        downloaded_file = ydl.prepare_filename(info).replace(info['ext'], 'wav')
        title = info.get('title', 'Untitled Analysis')

        stream_info = ydl.extract_info(link, download=False)
        resolved_media_url = None
        if 'url' in stream_info:
            resolved_media_url = stream_info['url']
        elif 'formats' in stream_info:
            audio_formats = [f for f in stream_info['formats'] if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
            if audio_formats:
                best_audio = max(audio_formats, key=lambda f: f.get('abr', 0))
                resolved_media_url = best_audio['url']
            else: 
                resolved_media_url = stream_info['formats'][0]['url']

    return downloaded_file, resolved_media_url, title

