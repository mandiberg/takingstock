'''
get list of all mp4 files in ROOT folder
Use moviepy to open each video file
concatenate the videos, with a fade to black transition that is FADE_DURATION seconds long
'''
import os
import sys

try:
    from moviepy import * # Simple and nice, the __all__ is set in moviepy so only useful things will be loaded
    from moviepy import VideoFileClip # You can also import only the things you really need
    from moviepy.video.fx import FadeIn, FadeOut

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

# ROOT = "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/takingstock_dropbox/MUD_model_files"
ROOT = "/Volumes/OWC52/heft_loop_scratch/60s"
FADE_DURATION = .5  # seconds

OUTPUT_FOLDER = os.path.join(ROOT, "concat_videos")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
combined_clips = None
for filename in os.listdir(ROOT):
    if filename.endswith(".mp4"):
        input_filepath = os.path.join(ROOT, filename)
        print(f"Processing video file: {input_filepath}")
        clip = VideoFileClip(input_filepath)
        # add fade out to black at the end of the clip
        clip = clip.with_effects([FadeOut(FADE_DURATION, 0)])
        #add fade in from black at the start of the clip
        clip = clip.with_effects([FadeIn(FADE_DURATION, 0)])
        # add the clip to the combined_clips,
        if combined_clips is None:
            combined_clips = clip
        else:
            combined_clips = concatenate_videoclips([combined_clips, clip])

# final_clip = concatenate_videoclips(combined_clips)
output_filepath = os.path.join(OUTPUT_FOLDER, "concatenated_video.mp4")
combined_clips.write_videofile(output_filepath, codec="libx264")
print(f"Saved concatenated video to: {output_filepath}")