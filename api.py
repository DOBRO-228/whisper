import os
import openai
import argparse
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

# Set your OpenAI API key
client = openai.OpenAI()

# Constants
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB in bytes


# Step 1: Extract audio from video file
def extract_audio(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', bitrate='192k')
    video_clip.close()


# Step 2: Split audio into parts less than 25MB
def split_audio(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    parts = []

    # Calculate how many bytes are roughly 25MB
    audio_length_ms = len(audio)
    part_size = int((MAX_FILE_SIZE / audio.frame_rate / audio.frame_width / audio.channels) * 1000)

    start = 0
    while start < audio_length_ms:
        end = min(start + part_size, audio_length_ms)
        parts.append(audio[start:end])
        start = end

    return parts


# Step 3: Transcribe each part using OpenAI's Whisper API
def transcribe_audio(audio_segment, index):
    temp_path = f"temp_part_{index}.wav"
    audio_segment.export(temp_path, format="wav", bitrate='192k')
    with open(temp_path, "rb") as audio_file:
        response = client.Audio.transcribe("whisper-1", audio_file)
    os.remove(temp_path)
    return response['text']


# Main function to combine all steps
def main():
    parser = argparse.ArgumentParser(description="Transcribe audio from a video file using OpenAI's Whisper API.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_text_path", type=str, help="Path to the output text file.")
    args = parser.parse_args()

    video_path = args.video_path
    output_text_path = args.output_text_path
    audio_path = "extracted_audio.wav"

    extract_audio(video_path, audio_path)
    audio_parts = split_audio(audio_path)

    with open(output_text_path, "w", encoding="utf-8") as out_file:
        number_of_parts = len(audio_parts)
        for index, part in enumerate(audio_parts):
            transcription = transcribe_audio(part, index)
            out_file.write(transcription + "\n")
            print(f"Part {index + 1} from {number_of_parts} transcribed.")

    if os.path.exists(audio_path):
        os.remove(audio_path)


if __name__ == "__main__":
    main()