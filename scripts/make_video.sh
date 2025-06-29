#!/bin/bash
# Script to create a 1-hour video from a short mp4 and mp3 file by looping both

set -e

# Settings
VIDEO=$(ls assets/*.mp4 2>/dev/null | head -n 1)
AUDIO=$(ls assets/*.mp3 2>/dev/null | head -n 1)
DURATION=3600  # 1 hour in seconds
OUTPUT="output1hour.mp4"

if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
  echo "❌ Video file not found in assets/."
  exit 1
fi

if [ -z "$AUDIO" ] || [ ! -f "$AUDIO" ]; then
  echo "❌ Audio file not found in assets/."
  exit 1
fi

# Get video duration in seconds (rounded down)
VIDEO_SECONDS=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO" | awk '{print int($1)}')
if [ -z "$VIDEO_SECONDS" ] || [ "$VIDEO_SECONDS" -le 0 ]; then
  echo "❌ Failed to determine video duration."
  exit 1
fi

# Calculate loop count needed
LOOPS=$(( ($DURATION + $VIDEO_SECONDS - 1) / $VIDEO_SECONDS ))

echo "🔁 Preparing concat list for $LOOPS loops of video..."
CONCAT_LIST="mylist.txt"
rm -f "$CONCAT_LIST"
for i in $(seq 1 $LOOPS); do
  echo "file '$PWD/$VIDEO'" >> "$CONCAT_LIST"
done

echo "🎞️ Concatenating video to create long video..."
ffmpeg -y -f concat -safe 0 -i "$CONCAT_LIST" -c copy long_video.mp4

echo "✂️ Trimming long video to exactly $DURATION seconds..."
ffmpeg -y -i long_video.mp4 -t $DURATION -c copy temp_video.mp4

echo "🎧 Looping and trimming audio to $DURATION seconds..."
ffmpeg -y -stream_loop -1 -i "$AUDIO" -t $DURATION \
  -af "afade=t=in:st=0:d=3,afade=t=out:st=$(($DURATION - 3)):d=3" \
  -c:a aac -b:a 192k temp_audio.aac

echo "🎬 Merging video and audio into final output..."
ffmpeg -y -i temp_video.mp4 -i temp_audio.aac \
  -c:v libx264 -c:a aac -shortest -pix_fmt yuv420p -movflags +faststart "$OUTPUT"

# Cleanup
rm -f "$CONCAT_LIST" long_video.mp4 temp_video.mp4 temp_audio.aac

echo "✅ Done! 1-hour video saved as: $OUTPUT"
