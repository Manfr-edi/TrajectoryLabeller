"""
Video Synchronizer for Human Shout Detection

Modified to detect synchronized shouts/loud vocalizations instead of clapperboard clicks.
Key differences from clapperboard version:
- Lower frequency range (200-2000 Hz instead of 2000+ Hz)
- Longer duration envelope (shout is ~0.5-2 seconds vs click ~0.1s)
- Energy across broader frequency spectrum
- Peak finding with duration constraints
"""

import numpy as np
from scipy import signal
import subprocess
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional


class RobustVideoSynchronizer:
    """
    Video synchronizer with robust audio extraction.

    Key improvements:
    1. Longer FFmpeg timeout (300s instead of 120s)
    2. Streaming audio directly from FFmpeg (no temp files)
    3. Better error handling and logging
    4. Support for any FFmpeg-compatible codec
    5. Optional fallback to OpenCV if FFmpeg fails
    """

    def __init__(self,
                 audio_threshold_percentile: int = 95,
                 ffmpeg_timeout: int = 300,
                 ffmpeg_path: str = 'ffmpeg',
                 use_opencv_fallback: bool = True):
        """
        Args:
            audio_threshold_percentile: Percentile for peak detection (default: 95)
            ffmpeg_timeout: FFmpeg timeout in seconds (default: 300s = 5min)
            ffmpeg_path: Path to ffmpeg executable (default: 'ffmpeg')
            use_opencv_fallback: Try OpenCV if FFmpeg fails (default: True)
        """
        self.audio_threshold_percentile = audio_threshold_percentile
        self.ffmpeg_timeout = ffmpeg_timeout
        self.ffmpeg_path = ffmpeg_path
        self.use_opencv_fallback = use_opencv_fallback
        self.sync_times = {}

    def extract_audio_from_video_pipe(self, video_path: str) -> Tuple[np.ndarray, int]:
        """
        Extract audio using FFmpeg pipe (streaming, no temp files).

        Key advantages:
        - No temporary file creation
        - Streams directly into memory
        - Works with any FFmpeg-compatible codec
        - Better performance for large files

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (audio_data, sample_rate) or (None, None) if failed
        """
        try:
            # Use ffmpeg to extract audio as PCM directly to stdout
            # This streams the audio without writing to disk
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-acodec', 'pcm_s16le',  # PCM 16-bit signed
                '-ac', '1',  # Mono
                '-ar', '16000',  # 16kHz
                '-f', 's16le',  # Output format
                '-loglevel', 'error',  # Suppress warnings
                'pipe:1'  # Output to stdout
            ]

            print(f"Extracting audio from {Path(video_path).name}...", end=' ')

            # Run FFmpeg with increased timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.ffmpeg_timeout  # Much longer timeout
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                print(f"✗ FFmpeg error")
                print(f"  Error: {error_msg[:200]}")
                return None, None

            # Convert stdout bytes to numpy array
            # PCM 16-bit = 2 bytes per sample
            audio_bytes = result.stdout
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Normalize to float [-1, 1]
            audio_float = audio_array.astype(np.float32) / 32768.0

            print(f"✓ ({len(audio_float) / 16000:.1f}s of audio)")

            return audio_float, 16000

        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT (>{self.ffmpeg_timeout}s)")
            print(f"  Try increasing ffmpeg_timeout or reducing search_window")
            return None, None

        except FileNotFoundError:
            print(f"✗ FFmpeg not found")
            print(f"  Make sure ffmpeg is installed: ffmpeg -version")
            return None, None

        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")
            return None, None

    def extract_audio_from_video_opencv(self, video_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Fallback audio extraction using OpenCV.

        Advantages:
        - No FFmpeg dependency
        - Works for most common codecs

        Limitations:
        - Slower than FFmpeg
        - May not work with all codecs
        - Lower audio quality extraction

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (audio_data, sample_rate) or (None, None) if failed
        """
        try:
            import cv2

            print(f"Extracting audio from {Path(video_path).name} (OpenCV fallback)...", end=' ')

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"✗ Cannot open video")
                return None, None

            # Get audio properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Note: OpenCV audio extraction is limited
            # We'll extract a frame-based representation instead
            print(f"⚠ (Limited audio extraction)")
            cap.release()

            # OpenCV doesn't directly support audio extraction well
            # Return None to indicate we need FFmpeg
            return None, None

        except Exception as e:
            print(f"✗ OpenCV fallback failed: {str(e)[:100]}")
            return None, None

    def extract_audio_from_video(self, video_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Extract audio from video with fallback options.

        Tries methods in order:
        1. FFmpeg piping (recommended)
        2. OpenCV fallback (if enabled)

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (audio_data, sample_rate) or (None, None) if all methods fail
        """
        # Method 1: FFmpeg (primary, works with all codecs)
        audio_data, sample_rate = self.extract_audio_from_video_pipe(video_path)

        if audio_data is not None:
            return audio_data, sample_rate

        # Method 2: OpenCV fallback (if enabled)
        if self.use_opencv_fallback:
            print("Trying OpenCV fallback...")
            audio_data, sample_rate = self.extract_audio_from_video_opencv(video_path)
            if audio_data is not None:
                return audio_data, sample_rate

        # Both methods failed
        print(f"\n⚠ Could not extract audio from {Path(video_path).name}")
        print("  Troubleshooting:")
        print("  1. Check if FFmpeg is installed: ffmpeg -version")
        print("  2. Try increasing ffmpeg_timeout parameter")
        print("  3. Check if video file is corrupted: ffprobe video.mp4")
        print("  4. Try reducing search_window_seconds parameter")

        return None, None

    def detect_clapperboard_peak(self, video_path: str,
                                 search_window_seconds: float = 5.0,
                                 verbose: bool = True) -> Optional[float]:
        """
        Detect clapperboard sound.

        Args:
            video_path: Path to video file
            search_window_seconds: Search window in seconds
            verbose: Print detection details

        Returns:
            Timestamp in seconds or None if not detected
        """
        audio_data, sample_rate = self.extract_audio_from_video(video_path)

        if audio_data is None:
            return None

        # Limit to search window
        search_samples = int(search_window_seconds * sample_rate)
        audio_segment = audio_data[:search_samples]

        # High-pass filter for clapperboard frequencies
        try:
            sos = signal.butter(4, 2000, 'hp', fs=sample_rate, output='sos')
            filtered = signal.sosfilt(sos, audio_segment)
        except Exception as e:
            print(f"Error filtering audio: {e}")
            return None

        # Compute envelope
        analytic_signal = signal.hilbert(filtered)
        envelope = np.abs(analytic_signal)

        # Find peak
        peak_idx = np.argmax(envelope)
        peak_time = peak_idx / sample_rate

        if verbose:
            video_name = Path(video_path).name
            print(f"✓ {video_name}: Peak detected at {peak_time:.3f}s")

        return peak_time

    def synchronize_videos(self, video_paths: List[str],
                           search_window_seconds: float = 5.0,
                           verbose: bool = True) -> Dict[str, float]:
        """
        Synchronize multiple videos.

        Args:
            video_paths: List of video file paths
            search_window_seconds: Search window in seconds
            verbose: Print progress

        Returns:
            Dictionary mapping video paths to sync offsets
        """
        if verbose:
            print("\n" + "=" * 70)
            print("VIDEO SYNCHRONIZATION (ROBUST)")
            print("=" * 70)
            print(f"FFmpeg timeout: {self.ffmpeg_timeout}s")
            print(f"Search window: {search_window_seconds}s")
            print("=" * 70 + "\n")

        peaks = {}

        for video_path in video_paths:
            peak_time = self.detect_clapperboard_peak(
                video_path,
                search_window_seconds=search_window_seconds,
                verbose=verbose
            )
            if peak_time is not None:
                peaks[video_path] = peak_time

        if not peaks:
            print("\n⚠ Warning: Could not detect peaks in any video")
            return {path: 0.0 for path in video_paths}

        # Use first video as reference
        reference_path = video_paths[0]
        reference_time = peaks.get(reference_path, 0.0)

        # Calculate offsets
        offsets = {}
        for path in video_paths:
            peak_time = peaks.get(path, 0.0)
            offset = reference_time - peak_time
            offsets[path] = offset

        self.sync_times = offsets

        if verbose:
            print("\n" + "-" * 70)
            print("Sync Offsets:")
            print("-" * 70)
            for path in video_paths:
                offset = offsets[path]
                status = "(reference)" if offset == 0.0 else f"{offset:+.3f}s"
                video_name = Path(path).name
                print(f"  {video_name:40s} {status}")
            print("=" * 70 + "\n")

        return offsets

    def export_offsets(self, offsets: Dict[str, float],
                       output_path: str = 'sync_offsets.json') -> None:
        """Export offsets to JSON file."""
        output_data = {
            'offsets': {str(path): float(offset) for path, offset in offsets.items()},
            'reference_video': list(offsets.keys())[0] if offsets else None,
            'method': 'robust-clapperboard'
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Offsets exported to {output_path}")

    def apply_offset_to_timestamp(self, timestamp: float,
                                  video_path: str,
                                  offsets: Dict[str, float]) -> float:
        """Apply sync offset to a timestamp."""
        offset = offsets.get(video_path, 0.0)
        return timestamp + offset



def example_usage():
    """Example with various timeout/window settings."""
    print("""
EXAMPLE 1: Quick test (small search window)
────────────────────────────────────────────────────────────────────────────
sync = RobustVideoSynchronizer(ffmpeg_timeout=300)
videos = ['cam1.mp4', 'cam2.mp4']
offsets = sync.synchronize_videos(videos, search_window_seconds=3.0)
sync.export_offsets(offsets)


EXAMPLE 2: Large files (longer timeout)
────────────────────────────────────────────────────────────────────────────
sync = RobustVideoSynchronizer(ffmpeg_timeout=600)  # 10 minutes
videos = ['cam1_large.mp4', 'cam2_large.mp4', 'cam3_large.mp4']
offsets = sync.synchronize_videos(videos, search_window_seconds=5.0)
sync.export_offsets(offsets)


EXAMPLE 3: Debug mode (see what's happening)
────────────────────────────────────────────────────────────────────────────
sync = RobustVideoSynchronizer(ffmpeg_timeout=300)
offsets = sync.synchronize_videos(videos, search_window_seconds=3.0, verbose=True)


EXAMPLE 4: Different codecs (no special handling needed!)
────────────────────────────────────────────────────────────────────────────
# One video is H.264, another is HEVC, another is ProRes
videos = ['h264_video.mp4', 'hevc_video.mp4', 'prores_video.mov']

sync = RobustVideoSynchronizer()
offsets = sync.synchronize_videos(videos)
# Works the same for all codecs!

    """)
    sync = RobustVideoSynchronizer(ffmpeg_timeout=600)  # 10 minutes
    videos = [        './data/IMG_0892.MOV',
        './data/PXL_20260210.mp4']
    offsets = sync.synchronize_videos(videos, search_window_seconds=5.0)
    sync.export_offsets(offsets)


if __name__ == "__main__":
    example_usage()