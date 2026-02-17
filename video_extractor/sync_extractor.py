import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from video_sync import RobustVideoSynchronizer
import time


class SynchronizedFrameExtractor:

    def __init__(self, ffmpeg_timeout: int = 600, max_workers: int = 2):
        self.ffmpeg_timeout = ffmpeg_timeout
        self.max_workers = max_workers
        self.synchronizer = RobustVideoSynchronizer(ffmpeg_timeout=ffmpeg_timeout)

    # ------------------------------------------------------------
    # 1️⃣  FORCE CFR NORMALIZATION (CRITICAL FIX)
    # ------------------------------------------------------------
    def convert_to_cfr(self, input_path: str, output_path: str, fps: int = 60):
        print(f"Converting {Path(input_path).name} → CFR {fps}fps...")

        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vf", f"fps={fps}",
            "-vsync", "cfr",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-y",
            output_path
        ]

        subprocess.run(cmd, check=True)
        print("✓ CFR conversion complete\n")

    def normalize_videos(self, video_paths: List[str], temp_dir="../cfr_videos"):
        Path(temp_dir).mkdir(exist_ok=True)

        normalized = []
        for path in video_paths:
            out_path = str(Path(temp_dir) / (Path(path).stem + "_CFR.mp4"))
            self.convert_to_cfr(path, out_path)
            normalized.append(out_path)

        return normalized

    # ------------------------------------------------------------
    # 2️⃣  FIXED FRAME EXTRACTION (NO FPS FILTER!)
    # ------------------------------------------------------------
    def extract_frames_from_video(
            self,
            video_path: str,
            output_dir: str,
            start_time: float,
            duration: float,
            frame_format: str = "png",
    ):

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem
        frame_pattern = os.path.join(output_dir, f"{video_name}_%06d.{frame_format}")

        cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-vsync", "0",
            "-q:v", "2",
            "-y",
            "-loglevel", "error",
            frame_pattern
        ]

        start = time.time()
        result = subprocess.run(cmd, capture_output=True, timeout=self.ffmpeg_timeout)
        elapsed = time.time() - start

        if result.returncode == 0:
            frames = list(Path(output_dir).glob(f"{video_name}_*.{frame_format}"))
            return len(frames), elapsed

        return 0, elapsed

    # ------------------------------------------------------------
    # 3️⃣  CORRECT OFFSET HANDLING
    # ------------------------------------------------------------
    def compute_start_time(self, offset: float) -> float:
        """
        If video was delayed (negative offset) → skip
        If early or reference → start at 0
        """
        if offset < 0:
            return -offset
        return 0.0

    # ------------------------------------------------------------
    # 4️⃣  FULL WORKFLOW
    # ------------------------------------------------------------
    def full_workflow(
            self,
            video_paths: List[str],
            output_base_dir="../synchronized_frames",
            duration: Optional[float] = None,
            frame_format="jpeg",
            verbose=True,
    ):

        # Use absolute paths to avoid key mismatch
        video_paths = [str(Path(p).resolve()) for p in video_paths]

        print("\nSTEP 1: Detecting initial clap\n")

        offsets = self.synchronizer.synchronize_videos(
            video_paths,
            search_window_seconds=15.0,
            verbose=True
        )

        print("\nSTEP 2: Extracting synchronized frames\n")

        # Compute common overlapping duration
        min_duration = float("inf")

        durations = {}

        for vid in video_paths:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    vid,
                ],
                capture_output=True,
                text=True,
            )

            total_duration = float(probe.stdout.strip())
            offset = offsets.get(vid, 0.0)

            start_time = -offset if offset < 0 else 0.0
            available = total_duration - start_time

            durations[vid] = (start_time, available)
            min_duration = min(min_duration, available)

        if duration:
            duration = min(duration, min_duration)
        else:
            duration = min_duration

        print(f"Common duration: {duration:.3f}s\n")

        # Extract frames
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

            futures = {}

            for idx, vid in enumerate(video_paths):
                offset = offsets.get(vid, 0.0)
                start_time = -offset if offset < 0 else 0.0

                output_dir = os.path.join(output_base_dir, f"cam{idx}")

                future = executor.submit(
                    self.extract_frames_from_video,
                    vid,
                    output_dir,
                    start_time,
                    duration,
                    frame_format,
                )

                futures[future] = vid

            for future in as_completed(futures):
                vid = futures[future]
                num_frames, elapsed = future.result()
                print(f"{Path(vid).name}: {num_frames} frames ({elapsed:.1f}s)")

        print("\n✓ Frames extracted")
        print("✓ Same number expected across cameras")
        print("✓ No drift expected if CFR videos were used\n")


if __name__ == "__main__":
    original_videos = [
        "../data/IMG_0892.MOV",
        "../data/PXL_20260210.mp4"
    ]

    videos = [
        "../cfr_videos/IMG_0892_CFR.mp4",
        "../cfr_videos/PXL_20260210_CFR.mp4"
    ]

    extractor = SynchronizedFrameExtractor()
    # UNCOMMENT IF YOU NEED TO NORMALIZE VIDEO FREQUENCY
    #normalized = extractor.normalize_videos(original_videos)
    extractor.full_workflow(videos, duration=10)
