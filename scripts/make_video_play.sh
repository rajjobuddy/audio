#!/bin/bash
set -e
DATE=$(date +"%d%m%y")
EPOCH=$(date +%s)

# Settings
VIDEO=$(ls assets/*.mp4 2>/dev/null | head -n 1)
AUDIO=$(ls assets/*.mp3 2>/dev/null | head -n 1)
DURATION=${1:-3600}   # default 1 hour
OUTPUT="${DATE}-${EPOCH}.mp4"

if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
  echo "❌ Video file not found in assets/."
  exit 1
fi

if [ -z "$AUDIO" ] || [ ! -f "$AUDIO" ]; then
  echo "❌ Audio file not found in assets/."
  exit 1
fi

echo "🎞️ Looping video (no concat, fast)..."
ffmpeg -y -stream_loop -1 -i "$VIDEO" -t $DURATION -c copy temp_video.mp4

echo "🎧 Looping and trimming audio..."
ffmpeg -y -stream_loop -1 -i "$AUDIO" -t $DURATION \
  -af "afade=t=in:st=0:d=3,afade=t=out:st=$(($DURATION - 3)):d=3" \
  -c:a aac -b:a 192k temp_audio.aac

echo "🎬 Merging video and audio..."
ffmpeg -i temp_video.mp4 -i temp_audio.aac -c:v libx264 -crf 25 -preset medium -movflags +frag_keyframe+empty_moov+faststart -c:a aac -b:a 128k output.mp4


# Cleanup
rm -f temp_video.mp4 temp_audio.aac

echo "✅ Done! Video saved as: $OUTPUT"
