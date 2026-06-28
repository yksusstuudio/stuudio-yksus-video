#!/usr/bin/env python3
"""
Stuudio Üksus Video Automation Backend
Handles: video editing, AI metadata generation, text overlays, TikTok prep
Deploy on: Replit or Railway
"""

import os
import sys
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
import logging

# Video/Audio processing
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.VideoClip import VideoClip
import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# Google Drive API
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# Claude API
from anthropic import Anthropic

# TTS (Google Cloud)
from google.cloud import texttospeech

# Utilities
from datetime import datetime
import ffmpeg

# ============================================================================
# SETUP
# ============================================================================

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
CONFIG = {
    'APPROVAL_MODE': os.getenv('APPROVAL_MODE', 'manual'),  # 'auto' or 'manual'
    'ADD_BRANDING': os.getenv('ADD_BRANDING', 'true').lower() == 'true',
    'Estonian_TTS': os.getenv('ESTONIAN_TTS', 'false').lower() == 'true',
    'MIN_VIDEO_LENGTH': int(os.getenv('MIN_VIDEO_LENGTH', '10')),  # seconds
    'MAX_VIDEO_LENGTH': int(os.getenv('MAX_VIDEO_LENGTH', '60')),  # seconds
    'SILENCE_THRESHOLD': float(os.getenv('SILENCE_THRESHOLD', '0.02')),  # audio threshold
    'MIN_SILENCE_DURATION': float(os.getenv('MIN_SILENCE_DURATION', '1.0')),  # seconds to cut
    'OUTPUT_QUALITY': os.getenv('OUTPUT_QUALITY', '720p'),  # 720p or 1080p
    'GOOGLE_DRIVE_SOURCE_FOLDER': os.getenv('GOOGLE_DRIVE_SOURCE_FOLDER'),
    'GOOGLE_DRIVE_OUTPUT_FOLDER': os.getenv('GOOGLE_DRIVE_OUTPUT_FOLDER'),
    'GOOGLE_DRIVE_APPROVED_FOLDER': os.getenv('GOOGLE_DRIVE_APPROVED_FOLDER'),
    'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
    'GOOGLE_CLOUD_PROJECT': os.getenv('GOOGLE_CLOUD_PROJECT'),
    'STUDIO_LOGO_PATH': os.getenv('STUDIO_LOGO_PATH', './uksus_logo.png'),
    'STUDIO_NAME': os.getenv('STUDIO_NAME', 'STUUDIO ÜKSUS'),
    'STUDIO_WEBSITE': os.getenv('STUDIO_WEBSITE', 'stuudioyksus.ee'),
}

# ============================================================================
# GOOGLE DRIVE HELPERS
# ============================================================================

class GoogleDriveManager:
    def __init__(self):
        self.service = None
        self.authenticate()

    def authenticate(self):
        """Authenticate with Google Drive API"""
        try:
            # Try service account first (for production)
            if os.path.exists('service_account.json'):
                credentials = Credentials.from_service_account_file(
                    'service_account.json',
                    scopes=['https://www.googleapis.com/auth/drive']
                )
            else:
                # Fallback: OAuth2 (for development)
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json',
                    scopes=['https://www.googleapis.com/auth/drive']
                )
                credentials = flow.run_local_server(port=0)
            
            self.service = build('drive', 'v3', credentials=credentials)
            logger.info("✓ Google Drive authenticated")
        except Exception as e:
            logger.error(f"✗ Google Drive auth failed: {e}")
            sys.exit(1)

    def list_files_in_folder(self, folder_id, mimetype='video/mp4'):
        """List files in a Drive folder"""
        try:
            query = f"'{folder_id}' in parents and mimeType='{mimetype}' and trashed=false"
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, createdTime)',
                pageSize=10
            ).execute()
            return results.get('files', [])
        except Exception as e:
            logger.error(f"✗ Failed to list files: {e}")
            return []

    def download_file(self, file_id, filename):
        """Download file from Drive"""
        try:
            request = self.service.files().get_media(fileId=file_id)
            with open(filename, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            logger.info(f"✓ Downloaded: {filename}")
            return True
        except Exception as e:
            logger.error(f"✗ Download failed: {e}")
            return False

    def upload_file(self, local_path, folder_id, filename=None):
        """Upload file to Drive folder"""
        try:
            if not filename:
                filename = os.path.basename(local_path)
            
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            media = MediaFileUpload(local_path, mimetype='video/mp4')
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            logger.info(f"✓ Uploaded: {filename} (ID: {file.get('id')})")
            return file.get('id')
        except Exception as e:
            logger.error(f"✗ Upload failed: {e}")
            return None

# ============================================================================
# VIDEO EDITING ENGINE
# ============================================================================

class VideoEditor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = VideoFileClip(video_path)
        self.duration = self.video.duration
        self.fps = self.video.fps
        logger.info(f"✓ Loaded video: {self.duration:.1f}s @ {self.fps}fps")

    def detect_silence(self, threshold=0.02, min_duration=1.0):
        """
        Detect silence segments in audio
        Returns list of (start, end) tuples for silent sections
        """
        try:
            if self.video.audio is None:
                logger.warning("⚠ No audio track found")
                return []

            # Load audio
            audio_array = self.video.audio.to_soundarray()
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Normalize
            audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-8)
            
            # Detect silence (RMS below threshold)
            frame_length = 2048
            hop_length = 512
            S = librosa.stft(audio_array)
            magnitude = np.abs(S)
            rms = librosa.feature.rms(S=magnitude, hop_length=hop_length)[0]
            
            # Find silent frames
            silent_frames = rms < threshold
            
            # Convert frame indices to time
            times = librosa.frames_to_time(np.arange(len(silent_frames)), sr=self.video.fps, hop_length=hop_length)
            
            # Find contiguous silent sections
            silence_segments = []
            in_silence = False
            silence_start = 0
            
            for i, is_silent in enumerate(silent_frames):
                if is_silent and not in_silence:
                    silence_start = times[i]
                    in_silence = True
                elif not is_silent and in_silence:
                    silence_duration = times[i] - silence_start
                    if silence_duration >= min_duration:
                        silence_segments.append((silence_start, times[i]))
                    in_silence = False
            
            if in_silence:
                silence_duration = self.duration - silence_start
                if silence_duration >= min_duration:
                    silence_segments.append((silence_start, self.duration))
            
            logger.info(f"✓ Detected {len(silence_segments)} silent sections")
            return silence_segments
        except Exception as e:
            logger.error(f"✗ Silence detection failed: {e}")
            return []

    def remove_silence(self, silence_segments):
        """
        Remove silent segments from video
        Returns edited VideoFileClip
        """
        if not silence_segments:
            return self.video
        
        try:
            # Build list of non-silent clips
            clips = []
            current_time = 0
            
            for silence_start, silence_end in sorted(silence_segments):
                if current_time < silence_start:
                    clips.append(self.video.subclip(current_time, silence_start))
                current_time = silence_end
            
            # Add final segment
            if current_time < self.duration:
                clips.append(self.video.subclip(current_time, self.duration))
            
            if clips:
                edited = concatenate_videoclips(clips)
                logger.info(f"✓ Removed silence: {self.duration:.1f}s → {edited.duration:.1f}s")
                return edited
            else:
                return self.video
        except Exception as e:
            logger.error(f"✗ Silence removal failed: {e}")
            return self.video

    def trim_to_length(self, video, min_length=10, max_length=60):
        """
        Trim video to target length
        If too long: speed up or trim from end
        If too short: keep as is
        """
        if video.duration <= max_length:
            logger.info(f"✓ Video length OK: {video.duration:.1f}s")
            return video
        
        # Too long: speed up
        speedup_factor = video.duration / max_length
        if speedup_factor > 1.5:
            # Too aggressive, trim instead
            trimmed = video.subclip(0, max_length)
            logger.info(f"✓ Trimmed to {max_length}s")
            return trimmed
        else:
            # Speed up
            sped = video.speedx(speedup_factor)
            logger.info(f"✓ Sped up {speedup_factor:.2f}x to {max_length}s")
            return sped

    def add_branding(self, video, logo_path=None, studio_name='STUUDIO ÜKSUS', website='stuudioyksus.ee'):
        """Add logo watermark + studio info overlay"""
        try:
            if not CONFIG['ADD_BRANDING']:
                return video
            
            # Add studio name + website text at bottom (last 3 seconds)
            txt_clip = TextClip(
                f"{studio_name}\n{website}",
                fontsize=24,
                color='white',
                font='Arial-Bold',
                method='caption',
                size=(video.w - 40, None),
                bg_color='black',
                align='center'
            ).set_position(('center', 'bottom')).set_duration(3).set_start(video.duration - 3)
            
            final = CompositeVideoClip([video, txt_clip])
            logger.info(f"✓ Added branding")
            return final
        except Exception as e:
            logger.error(f"✗ Branding failed: {e}")
            return video

    def add_text_overlay(self, video, text, duration=None):
        """Add text overlay (caption)"""
        try:
            if duration is None:
                duration = min(5, video.duration)
            
            txt = TextClip(
                text,
                fontsize=32,
                color='white',
                font='Arial-Bold',
                method='caption',
                size=(video.w - 40, None),
                bg_color='rgba(0,0,0,0.7)',
                align='center'
            ).set_position(('center', 'top')).set_duration(duration).set_start(0)
            
            final = CompositeVideoClip([video, txt])
            logger.info(f"✓ Added text overlay")
            return final
        except Exception as e:
            logger.error(f"✗ Text overlay failed: {e}")
            return video

    def export(self, output_path, quality='720p', codec='libx264'):
        """Export video"""
        try:
            # Quality settings
            bitrate_map = {
                '720p': '2500k',
                '1080p': '4500k',
                '480p': '1000k'
            }
            bitrate = bitrate_map.get(quality, '2500k')
            
            self.video.write_videofile(
                output_path,
                codec=codec,
                audio_codec='aac',
                bitrate=bitrate,
                verbose=False,
                logger=None
            )
            logger.info(f"✓ Exported: {output_path}")
            return True
        except Exception as e:
            logger.error(f"✗ Export failed: {e}")
            return False

    def close(self):
        """Clean up"""
        self.video.close()

# ============================================================================
# AI METADATA GENERATION
# ============================================================================

class MetadataGenerator:
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)

    def generate_metadata(self, video_filename, duration):
        """Generate caption, hashtags, CTA, category"""
        prompt = f"""
You are a social media expert for a calisthenics studio in Estonia (Stuudio Üksus).

Video filename: {video_filename}
Duration: {duration:.1f} seconds

Based on the filename and duration, generate:
1. A catchy, motivational caption (1-2 sentences, engaging for TikTok)
2. 7-10 relevant hashtags (#calisthenics, #strength, #bodyweight, etc.) in Estonian + English
3. A call-to-action (e.g., "Join us for a class!", "Build strength with us!")
4. Category: strength|skill|motivation|technique|challenge|transformation

Respond ONLY as JSON:
{{
  "caption": "string",
  "hashtags": ["#tag1", "#tag2", ...],
  "cta": "string",
  "category": "strength|skill|motivation|technique|challenge|transformation"
}}
"""
        try:
            response = self.client.messages.create(
                model='claude-opus-4-6',
                max_tokens=300,
                messages=[
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            # Parse JSON response
            text = response.content[0].text
            data = json.loads(text)
            logger.info(f"✓ Generated metadata: {data['category']}")
            return data
        except Exception as e:
            logger.error(f"✗ Metadata generation failed: {e}")
            return {
                'caption': 'Check out this calisthenics moment!',
                'hashtags': ['#calisthenics', '#strength', '#bodyweight', '#fitness'],
                'cta': 'Join us for a class!',
                'category': 'strength'
            }

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def process_video(video_id, video_name, drive_manager):
    """Main pipeline: download → edit → add metadata → upload"""
    
    temp_dir = './temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. Download
    local_path = os.path.join(temp_dir, video_name)
    if not drive_manager.download_file(video_id, local_path):
        return None
    
    # 2. Edit
    editor = VideoEditor(local_path)
    
    # Detect & remove silence
    silence_segments = editor.detect_silence(
        threshold=CONFIG['SILENCE_THRESHOLD'],
        min_duration=CONFIG['MIN_SILENCE_DURATION']
    )
    edited_video = editor.remove_silence(silence_segments)
    
    # Trim to length
    edited_video = editor.trim_to_length(
        edited_video,
        min_length=CONFIG['MIN_VIDEO_LENGTH'],
        max_length=CONFIG['MAX_VIDEO_LENGTH']
    )
    
    # 3. Generate metadata
    meta_gen = MetadataGenerator(CONFIG['ANTHROPIC_API_KEY'])
    metadata = meta_gen.generate_metadata(video_name, edited_video.duration)
    
    # 4. Add text overlay
    edited_video = editor.add_text_overlay(edited_video, metadata['caption'])
    
    # 5. Add branding
    edited_video = editor.add_branding(
        edited_video,
        logo_path=CONFIG['STUDIO_LOGO_PATH'],
        studio_name=CONFIG['STUDIO_NAME'],
        website=CONFIG['STUDIO_WEBSITE']
    )
    
    # 6. Export
    output_filename = f"{Path(video_name).stem}_edited.mp4"
    output_path = os.path.join(temp_dir, output_filename)
    
    if not editor.export(output_path, quality=CONFIG['OUTPUT_QUALITY']):
        editor.close()
        return None
    
    editor.close()
    
    # 7. Upload to Drive
    if CONFIG['APPROVAL_MODE'] == 'manual':
        output_folder = CONFIG['GOOGLE_DRIVE_OUTPUT_FOLDER']
    else:
        output_folder = CONFIG['GOOGLE_DRIVE_APPROVED_FOLDER']
    
    file_id = drive_manager.upload_file(output_path, output_folder, output_filename)
    
    # 8. Cleanup
    os.remove(local_path)
    os.remove(output_path)
    
    # 9. Return metadata for logging
    return {
        'source_file': video_name,
        'output_file': output_filename,
        'output_file_id': file_id,
        'duration_original': editor.duration,
        'duration_edited': edited_video.duration,
        'metadata': metadata,
        'processed_at': datetime.now().isoformat(),
        'approval_mode': CONFIG['APPROVAL_MODE']
    }

def main():
    """Run automation"""
    logger.info("=" * 60)
    logger.info("STUUDIO ÜKSUS VIDEO AUTOMATION ENGINE")
    logger.info(f"Mode: {CONFIG['APPROVAL_MODE'].upper()}")
    logger.info("=" * 60)
    
    # Validate config
    if not all([CONFIG['GOOGLE_DRIVE_SOURCE_FOLDER'], CONFIG['ANTHROPIC_API_KEY']]):
        logger.error("✗ Missing required env vars")
        sys.exit(1)
    
    # Init
    drive = GoogleDriveManager()
    
    # Get unprocessed videos
    files = drive.list_files_in_folder(CONFIG['GOOGLE_DRIVE_SOURCE_FOLDER'])
    if not files:
        logger.info("ℹ No new videos found")
        return []
    
    logger.info(f"Found {len(files)} video(s) to process")
    
    results = []
    for file in files:
        logger.info(f"\n→ Processing: {file['name']}")
        result = process_video(file['id'], file['name'], drive)
        if result:
            results.append(result)
            # Move source to archive (optional)
            # drive.move_file(file['id'], archive_folder_id)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ Processed {len(results)}/{len(files)} videos")
    logger.info("=" * 60)
    
    return results

if __name__ == '__main__':
    results = main()
    print(json.dumps(results, indent=2))
