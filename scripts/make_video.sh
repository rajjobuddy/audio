#!/bin/bash
DATE=$(date +"%d%m%y")

IMAGE="assets/image.jpg"
AUDIO="assets/music.mp3"
OUTPUT="${DATE}video10.mp4"
DURATION=600

if [ ! -f "$IMAGE" ]; then
  echo "❌ Image file not found: $IMAGE"
  exit 1
fi

if [ ! -f "$AUDIO" ]; then
  echo "❌ Audio file not found: $AUDIO"
  exit 1
fi

echo "🎞️ Creating still background video..."
ffmpeg -y -loop 1 -i "$IMAGE" -t $DURATION -vf "fade=t=in:st=0:d=3,fade=t=out:st=$(($DURATION - 3)):d=3,format=yuv420p" \
  -c:v libx264 -pix_fmt yuv420p temp_video.mp4

echo "🎧 Looping and trimming audio..."
ffmpeg -y -stream_loop 1000 -i "$AUDIO" -t $DURATION \
  -af "afade=t=in:st=0:d=3,afade=t=out:st=$(($DURATION - 3)):d=3" \
  -c:a aac -b:a 192k temp_audio.aac

echo "🖊️ Merging with text overlay..."
ffmpeg -y -i temp_video.mp4 -i temp_audio.aac -filter_complex "[0:v]drawtext=text='Relax | Calm | Sleep':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2" \
  -c:v libx264 -c:a aac -shortest -pix_fmt yuv420p -movflags +faststart "$OUTPUT"

rm temp_video.mp4 temp_audio.aac
echo "✅ Done! Video saved as: $OUTPUT"
