#!/bin/bash
DATE=$(date +"%d%m%y")
EPOCH=$(date +%s)

IMAGE=$(ls assets/*.jpg)
AUDIO=$(ls assets/*.mp3)
OUTPUT="${DATE}-${EPOCH}.mp4"
DURATION=${1:-14400}   # default 4 hours (14400 sec)

# Check files
[ ! -f "$IMAGE" ] && echo "❌ Image not found: $IMAGE" && exit 1
[ ! -f "$AUDIO" ] && echo "❌ Audio not found: $AUDIO" && exit 1

echo "🎞️ Creating still video (fast)..."
ffmpeg -y -loop 1 -framerate 1 -i "$IMAGE" -t $DURATION -c:v libx264 -pix_fmt yuv420p temp_video.mp4

echo "🎧 Looping audio (fast)..."
ffmpeg -y -stream_loop -1 -i "$AUDIO" -t $DURATION -c:a aac -b:a 192k temp_audio.aac

echo "🖊️ Merging video + audio..."
ffmpeg -y -i temp_video.mp4 -i temp_audio.aac -c:v copy -c:a copy -shortest "$OUTPUT"

rm temp_video.mp4 temp_audio.aac
echo "✅ Done! Video saved as: $OUTPUT"
