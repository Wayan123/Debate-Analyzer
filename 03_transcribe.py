import whisper
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="input mp3 path")
ap.add_argument("-o", "--output", required=True, help="output txt path")
args = vars(ap.parse_args())

model = whisper.load_model("medium")
input_audio_path = args["input"]
result = model.transcribe(input_audio_path)

transcribed_text = result["text"]
print(transcribed_text)

output_file_path = args["output"]

with open(output_file_path, "w", encoding="utf-8") as file:
    file.write(transcribed_text)

print(f"Transcribed text has been saved to '{output_file_path}'")
