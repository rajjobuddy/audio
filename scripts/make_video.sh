#!/bin/bash
DATE=$(date +"%d%m%y")
VIDEO=$(ls assets/*.mp4 2>/dev/null | head -n 1)
AUDIO=$(ls assets/*.mp3 2>/dev/null | head -n 1)
OUTPUT="${DATE}video10.mp4"
DURATION=${1:-900}

if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
  echo "❌ Video file not found in assets/."
  exit 1
fi

if [ -z "$AUDIO" ] || [ ! -f "$AUDIO" ]; then
  echo "❌ Audio file not found in assets/."
  exit 1
fi

echo "🎞️ Creating looped/truncated video..."
ffmpeg -y -stream_loop 1000 -i "$VIDEO" -t $DURATION \
  -vf "fade=t=in:st=0:d=3,fade=t=out:st=$(($DURATION - 3)):d=3,format=yuv420p" \
  -c:v libx264 -pix_fmt yuv420p temp_video.mp4

echo "🎧 Looping and trimming audio..."
ffmpeg -y -stream_loop 1000 -i "$AUDIO" -t $DURATION \
  -af "afade=t=in:st=0:d=3,afade=t=out:st=$(($DURATION - 3)):d=3" \
  -c:a aac -b:a 192k temp_audio.aac

echo "🖊️ Merging video and audio..."
ffmpeg -y -i temp_video.mp4 -i temp_audio.aac \
  -c:v libx264 -c:a aac -shortest -pix_fmt yuv420p -movflags +faststart "$OUTPUT"

rm temp_video.mp4 temp_audio.aac
echo "✅ Done! Video saved as: $OUTPUT"
