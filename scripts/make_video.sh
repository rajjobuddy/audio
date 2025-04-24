#!/bin/bash
DATE=$(date +"%d%m%y")

IMAGE=`ls assets/*.jpg`
AUDIO=`ls assets/*.mp3`
echo $IMAGE
echo $AUDIO
OUTPUT="${DATE}video10.mp4"
DURATION=${1:-3600}

ffprobe "$IMAGE"

if [ ! -f "$IMAGE" ]; then
  echo "❌ Image file not found: $IMAGE"
  exit 1
fi

if [ ! -f "$AUDIO" ]; then
  echo "❌ Audio file not found: $AUDIO"
  exit 1
fi
mv "$IMAGE" image.jpg
mv "$AUDIO" audio.mp3
echo "🎞️ Creating still background video..."
ffmpeg -y -i "image.jpg" -t $DURATION -vf "fade=t=in:st=0:d=3,fade=t=out:st=$(($DURATION - 3)):d=3,format=yuv420p" \
  -c:v libx264 -pix_fmt yuv420p temp_video.mp4

echo "🎧 Looping and trimming audio..."
ffmpeg -y -stream_loop 1000 -i "audio.mp3" -t $DURATION \
  -af "afade=t=in:st=0:d=3,afade=t=out:st=$(($DURATION - 3)):d=3" \
  -c:a aac -b:a 192k temp_audio.aac

echo "🖊️ Merging with text overlay..."
#ffmpeg -y -i temp_video.mp4 -i temp_audio.aac -filter_complex "[0:v]drawtext=text=' ':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2" \
#  -c:v libx264 -c:a aac -shortest -pix_fmt yuv420p -movflags +faststart "$OUTPUT"
ffmpeg -y -i temp_video.mp4 -i temp_audio.aac \
  -c:v libx264 -c:a aac -shortest -pix_fmt yuv420p -movflags +faststart "$OUTPUT"

rm temp_video.mp4
echo "✅ Done! Video saved as: $OUTPUT"
