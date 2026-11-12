"""Turn a zip of numbered PNG frames into an H.264 video, without unpacking the zip.

Made for the Bosch Small Traffic Lights test set (dataset_test_rgb.zip, 12 GB, 8334 frames,
about 15 fps). macOS `unzip` cannot read zips over 4 GB, so the frames are read with Python
and piped into ffmpeg.

Usage (from the repo root, ffmpeg must be installed):
  python src/video_pipeline/frames_to_video.py --zip datasets/dataset_test_rgb.zip --out datasets/videos/bosch_test.mp4
"""
import argparse
import subprocess
import zipfile
from pathlib import Path


def zip_frames_to_video(zip_path, out_path, fps=15):
    """Write every .png in the zip, in numeric order of the file name, to out_path. Return the frame count."""
    with zipfile.ZipFile(zip_path) as z:
        names = sorted((n for n in z.namelist() if n.endswith(".png")), key=lambda n: int(Path(n).stem))
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-c:v", "png", "-framerate", str(fps),
             "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out_path)],
            stdin=subprocess.PIPE)
        try:
            for n in names:
                ffmpeg.stdin.write(z.read(n))
            ffmpeg.stdin.close()
        except BaseException:
            # Without this, ffmpeg would finish a truncated video that looks valid (Ctrl-C, bad zip entry, full disk).
            ffmpeg.kill()
            ffmpeg.wait()
            Path(out_path).unlink(missing_ok=True)
            raise
    if ffmpeg.wait() != 0:
        Path(out_path).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed while writing {out_path}")
    return len(names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", required=True, help="zip file with numbered PNG frames")
    parser.add_argument("--out", required=True, help="output .mp4")
    parser.add_argument("--fps", type=float, default=15, help="frame rate of the frames (Bosch: about 15)")
    args = parser.parse_args()
    print(f"wrote {zip_frames_to_video(args.zip, args.out, args.fps)} frames to {args.out}")
