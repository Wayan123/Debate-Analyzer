from moviepy.editor import VideoFileClip
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="input mp4 path")
ap.add_argument("-o", "--output", required=True, help="output mp3 path")
args = vars(ap.parse_args())


def extract_audio_from_mp4(input_file, output_file):
    video_clip = VideoFileClip(input_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_file)


if __name__ == "__main__":
    input_file = args["video"]
    output_file = args["output"]
    extract_audio_from_mp4(input_file, output_file)
