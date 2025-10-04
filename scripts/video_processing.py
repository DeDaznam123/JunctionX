import yt_dlp
def extract_audio_from_link(link):
    ydl_opts = {
        'format': 'bestaudio/best',
        'skip_download': True,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=False)
    return info['url']

info = extract_audio_from_link("https://www.reddit.com/r/videos/comments/1nxgwfr/so_many_comedians_just_ruined_their_reputations/")
print(info)

