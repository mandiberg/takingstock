'''
get list of all mp4 files in ROOT folder
Use moviepy to open each video file
make a new video file that has the original video content looped through LOOPS_PER_VIDEO times
save that in a new subfolder called "looped_videos"
'''
import os
import sys
try:
    from moviepy import * # Simple and nice, the __all__ is set in moviepy so only useful things will be loaded
    from moviepy import VideoFileClip # You can also import only the things you really need
except ModuleNotFoundError as e:
    # Diagnostic help when moviepy is installed in a different environment
    print("ModuleNotFoundError:", e)
    print("Python executable:", sys.executable)
    print("To install into this Python run:")
    print(f"    {sys.executable} -m pip install moviepy imageio_ffmpeg")
    print("Or run this script with the Python that already has moviepy installed, e.g.:")
    print("    /path/to/env/bin/python loop_video_file.py")
    # Optionally show where moviepy is installed if importlib.metadata can find it
    try:
        import importlib.metadata as _md
        try:
            dist = _md.distribution("moviepy")
            print("moviepy (other env) files:", dist.locate_file(""))
        except Exception:
            pass
    except Exception:
        pass
    raise SystemExit("moviepy is not importable in this Python environment.")

ROOT = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/Mandiberg-Heft/Object_inventory/November7_samples"
ROOT = "/Volumes/OWC4/segment_images/renderfolder"
LOOPS_PER_VIDEO = 3

OUTPUT_FOLDER = os.path.join(ROOT, "looped_videos")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
for filename in os.listdir(ROOT):
    if filename.endswith(".mp4"):
        input_filepath = os.path.join(ROOT, filename)
        print(f"Processing video file: {input_filepath}")
        clip = VideoFileClip(input_filepath)
        clips = [clip] * LOOPS_PER_VIDEO
        final_clip = concatenate_videoclips(clips)
        output_filepath = os.path.join(OUTPUT_FOLDER, filename)
        final_clip.write_videofile(output_filepath, codec="libx264")
        print(f"Saved looped video to: {output_filepath}")