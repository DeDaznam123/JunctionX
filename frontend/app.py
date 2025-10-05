import streamlit as st
import requests
import pandas as pd
from streamlit_cookies_manager import EncryptedCookieManager
import json

# --- Page Config ---
st.set_page_config(
    page_title="Audio Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for a wider sidebar and bigger font
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 400px !important; /* Set the width to your desired value */
    }
    [data-testid="stTextInput"] input {
        font-size: 1.25rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Cookie Manager ---
# This should be on the top of your script
cookies = EncryptedCookieManager(
    password=st.secrets["COOKIE_PASSWORD"],
)

if not cookies.ready():
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.title("Extremism Screener")
    st.markdown("---")

    # --- History ---
    # The cache now stores {url: {'name': '...', 'analysis': ...}}
    cache_str = cookies.get('analysis_cache')
    analysis_cache = json.loads(cache_str) if cache_str else {}

    st.header("Analysis History")
    history_urls = list(analysis_cache.keys())

    if not history_urls:
        st.info("Your analyzed URLs will appear here.")
    else:
        # Display history using the custom name
        for url_item in history_urls:
            item_name = analysis_cache[url_item].get('name', url_item) # Fallback to URL if name doesn't exist
            if st.button(item_name, key=f"history_{url_item}"):
                st.session_state.analysis_results = analysis_cache[url_item]['analysis']
                st.session_state.current_url = url_item
                st.session_state.audio_start_time = 0
                st.rerun()

    if st.button("Clear History"):
        cookies['analysis_cache'] = json.dumps({})
        cookies.save()
        if 'analysis_results' in st.session_state:
            del st.session_state.analysis_results
        st.rerun()


# --- Main Page ---
st.markdown("<h2 style='text-align: center;'>Analyze Audio from URL</h2>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Enter a URL to a video or audio file (e.g., YouTube, Reddit, or a direct link). "
    "The system will extract the audio, clean it, and transcribe it.</p>",
    unsafe_allow_html=True
)

# Create a centered container with fixed max width
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Create two columns for input and button side by side
    input_col, button_col = st.columns([4, 1])
    with input_col:
        url = st.text_input("URL to analyze", "", key="url_input", label_visibility="collapsed")
    with button_col:
        analyze_button = st.button("Analyze", type="primary", use_container_width=True)

if analyze_button:
    if url:
        st.session_state.current_url = url # Keep track of the submitted URL
        if url in analysis_cache:
            st.session_state.analysis_results = analysis_cache[url]['analysis']
            st.session_state.audio_start_time = 0
            st.rerun()
        else:
            with st.spinner("Analyzing... This may take a few minutes depending on the audio length and model size."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/analyze",
                        params={"link": url, "strength": "light", "model_size": "small.en"}
                    )
                    response.raise_for_status()

                    data = response.json()
                    st.session_state.analysis_results = data
                    st.session_state.audio_start_time = 0

                    # --- Save to Cache with the fetched video title ---
                    video_title = data.get('video_title', f"Analysis - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
                    analysis_cache[url] = {'name': video_title, 'analysis': data}

                    if len(analysis_cache) > 10:
                        oldest_url = next(iter(analysis_cache))
                        del analysis_cache[oldest_url]

                    cookies['analysis_cache'] = json.dumps(analysis_cache)
                    cookies.save()

                    st.rerun()

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection Error: Could not connect to the backend. Is it running? \n\nDetails: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a URL to analyze.")

# --- Results Display ---
if 'analysis_results' in st.session_state:
    current_url = st.session_state.current_url

    # --- Editable Title ---
    current_name = analysis_cache.get(current_url, {}).get('name', "Analysis")
    new_name = st.text_input("Analysis Name", value=current_name, key=f"rename_{current_url}")

    # If name is changed, update the cache and cookie
    if new_name != current_name:
        analysis_cache[current_url]['name'] = new_name
        cookies['analysis_cache'] = json.dumps(analysis_cache)
        cookies.save()
        st.rerun() # Rerun to update the sidebar

    data = st.session_state.analysis_results

    # --- Original Link ---
    st.markdown(f"**URL:** [{current_url}]({current_url})")

    # --- Audio Player ---
    st.subheader("Audio")
    if data.get("resolved_media_url"):
        st.audio(data["resolved_media_url"], start_time=st.session_state.get('audio_start_time', 0))
    else:
        st.warning("No audio could be resolved from the URL.")

    # --- Editable Transcription List ---
    st.subheader("Transcription Segments")
    if 'transcription' in data and data['transcription']:

        segments_to_remove = []

        for i, segment in enumerate(data['transcription']):
            start_time = segment['start']
            text = segment['text']

            col1, col2, col3 = st.columns([2, 8, 2])

            with col1:
                if st.button(f"{pd.to_datetime(start_time, unit='s').strftime('%M:%S')}", key=f"play_{i}"):
                    st.session_state.audio_start_time = int(start_time)
                    st.rerun()

            with col2:
                st.markdown(f"> {text}")

            with col3:
                if st.button("Remove", key=f"remove_{i}"):
                    segments_to_remove.append(i)

        if segments_to_remove:
            for index in sorted(segments_to_remove, reverse=True):
                st.session_state.analysis_results['transcription'].pop(index)

            # --- Update Cache in Cookie after edit ---
            analysis_cache[current_url]['analysis'] = st.session_state.analysis_results
            cookies['analysis_cache'] = json.dumps(analysis_cache)
            cookies.save()
            st.rerun()

    else:
        st.warning("No transcription available.")

    # --- Expander for Raw Data ---
    with st.expander("Raw JSON Response"):
        st.json(st.session_state.analysis_results)



