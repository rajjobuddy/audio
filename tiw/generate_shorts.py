import os
import math
import time
from datetime import datetime
import json
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part

from google.oauth2 import service_account
from moviepy import AudioFileClip, ImageClip, concatenate_videoclips


# -----------------------
# CONFIGURATION (ENV-FIRST)
# -----------------------
SERVICE_ACCOUNT_KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
LOCATION = os.environ.get("LOCATION", "us-central1")

MP3_FILE = os.environ.get("MP3_FILE")  # workflow sets this (tiw/input/*.mp3 or *.mpeg)

IMAGE_DURATION = float(os.environ.get("IMAGE_DURATION", "5"))
ASPECT_RATIO = os.environ.get("ASPECT_RATIO", "9:16")

CHANNEL_NAME = os.environ.get("CHANNEL_NAME", "The Investment Wire")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "tiw/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, f"youtube_shorts_{timestamp}.mp4")
OUTPUT_METADATA = os.path.join(OUTPUT_DIR, f"youtube_metadata_{timestamp}.json")


# -----------------------
# INIT VERTEX AI
# -----------------------
def init_vertex_ai():
    if not SERVICE_ACCOUNT_KEY:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. "
            "Set it to the path of your service account JSON."
        )

    if not os.path.exists(SERVICE_ACCOUNT_KEY):
        raise FileNotFoundError(f"Missing credentials file at: {SERVICE_ACCOUNT_KEY}")

    with open(SERVICE_ACCOUNT_KEY, "r", encoding="utf-8") as f:
        sa_info = json.load(f)

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or sa_info.get("project_id")
    if not project_id:
        raise RuntimeError("Could not determine project_id from service account JSON.")

    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_KEY)
    vertexai.init(project=project_id, location=LOCATION, credentials=credentials)

    print(f"✓ Project ID: {project_id}")
    print(f"✓ Location: {LOCATION}")


# -----------------------
# HELPERS
# -----------------------
def get_audio_duration(file_path: str) -> float:
    try:
        audio = AudioFileClip(file_path)
        duration = float(audio.duration or 0.0)
        audio.close()
        return duration
    except Exception as e:
        print(f"Error reading audio '{file_path}': {e}")
        return 0.0


def _parse_mmss(s: str) -> float:
    # "0:05" -> 5.0
    try:
        m, sec = s.strip().split(":")
        return int(m) * 60 + float(sec)
    except Exception:
        return 0.0


def _format_mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}"


def _extract_timestamp_chunks(transcript: str):
    """
    Expected lines like:
    [0:00-0:05] text...
    """
    chunks = []
    pattern = re.compile(r"\[(\d+:\d+)\s*-\s*(\d+:\d+)\]\s*(.*)")
    for line in (transcript or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            continue
        start_s = _parse_mmss(m.group(1))
        end_s = _parse_mmss(m.group(2))
        text = (m.group(3) or "").strip()
        if end_s <= start_s:
            continue
        chunks.append({"start": start_s, "end": end_s, "text": text})
    return chunks


def _segments_from_chunks(chunks, audio_duration: float, target_n: int):
    """
    Build exactly target_n segments with (start,end,text) by time-windowing.
    If transcript parsing fails, falls back to uniform windows.
    """
    if target_n <= 0:
        return []

    if not chunks:
        seg_len = audio_duration / target_n if audio_duration > 0 else IMAGE_DURATION
        segs = []
        for i in range(target_n):
            st = i * seg_len
            en = min(audio_duration, (i + 1) * seg_len)
            if i == target_n - 1 and audio_duration > 0:
                en = audio_duration
            segs.append({"start": st, "end": en, "text": ""})
        return segs

    # Use uniform windows across audio, then pull transcript text that overlaps each window
    seg_len = audio_duration / target_n if audio_duration > 0 else IMAGE_DURATION
    segs = []
    for i in range(target_n):
        st = i * seg_len
        en = min(audio_duration, (i + 1) * seg_len)
        if i == target_n - 1 and audio_duration > 0:
            en = audio_duration

        texts = []
        for c in chunks:
            if c["end"] <= st or c["start"] >= en:
                continue
            if c["text"]:
                texts.append(c["text"])
        segs.append({"start": st, "end": en, "text": " ".join(texts).strip()})

    return segs


def transcribe_audio_with_timestamps(audio_part, model) -> str:
    transcript_prompt = (
        "Transcribe this audio with approximate timestamps ~5 seconds each. "
        "Format exactly per line: [0:00-0:05] text here"
    )
    transcript_response = model.generate_content([audio_part, transcript_prompt])
    return (transcript_response.text or "").strip()


def _clean_model_lines(text: str):
    lines = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # remove leading numbering like "1. "
        line = re.sub(r"^\d+\.\s*", "", line).strip()
        lines.append(line)
    return lines


def _fallback_overlay_english(segment_text: str) -> str:
    # If Gemini fails, still produce something in English
    if not segment_text:
        return "Key update"
    # crude, but guarantees English output format; better overlay comes from Gemini.
    return "Key update"


def generate_prompts_from_transcript_segments(segments, audio_part, model, num_images: int):
    """
    Returns:
      prompts_with_text: list[str] each like: "<image prompt> [TEXT: <english overlay>]"
    """
    # Build input for Gemini so it can make distinct prompts PER segment.
    seg_lines = []
    for i, seg in enumerate(segments):
        seg_lines.append(f"{i} ||| {_format_mmss(seg['start'])}-{_format_mmss(seg['end'])} ||| {seg.get('text','')}")
    seg_block = "\n".join(seg_lines)

    prompt_text = f"""
You are generating visuals for a YouTube Shorts video.

You will receive {num_images} segments, each containing:
index ||| timestamp ||| spoken text (may be Hindi).

Task:
- For EACH segment, create:
  1) A distinct image generation prompt that matches the meaning of that segment.
  2) An English text overlay (translate/summarize into English), 3 to 8 words max.

Output EXACTLY {num_images} lines.
Each line must be EXACTLY:
index ||| image_prompt ||| text_overlay_english

Rules:
- The image_prompt must be different for each segment (no repetition).
- text_overlay_english must ALWAYS be English.
- Avoid generic "stock market" unless the segment actually discusses markets.
- No extra commentary, no markdown.

Segments:
{seg_block}
"""

    resp = model.generate_content([audio_part, prompt_text])
    lines = _clean_model_lines(resp.text)

    by_idx = {}
    for line in lines:
        parts = [p.strip() for p in line.split("|||")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except Exception:
            continue
        img_prompt = parts[1]
        overlay_en = "|||".join(parts[2:]).strip()
        if img_prompt:
            by_idx[idx] = (img_prompt, overlay_en)

    prompts_with_text = []
    for i in range(num_images):
        if i in by_idx:
            img_prompt, overlay_en = by_idx[i]
            overlay_en = overlay_en.strip()
            if not overlay_en:
                overlay_en = _fallback_overlay_english(segments[i].get("text", ""))
            prompts_with_text.append(f"{img_prompt} [TEXT: {overlay_en}]")
        else:
            # fallback per segment: still unique-ish by including segment text
            base = segments[i].get("text", "").strip()
            img_prompt = (
                f"Vibrant cinematic American scene illustrating: {base[:160]}"
                if base else
                "Vibrant cinematic American scene, documentary style, high detail"
            )
            prompts_with_text.append(f"{img_prompt} [TEXT: {_fallback_overlay_english(base)}]")

    return prompts_with_text


def generate_prompts_and_transcript_from_audio(audio_path: str, num_images: int, audio_duration: float):
    print(f"Uploading and analyzing: {audio_path}...")

    model = GenerativeModel("gemini-2.0-flash-exp")

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    audio_part = Part.from_data(data=audio_data, mime_type="audio/mpeg")

    transcript = transcribe_audio_with_timestamps(audio_part, model)
    print(f"\n📝 Transcript:\n{transcript}\n")

    chunks = _extract_timestamp_chunks(transcript)
    segments = _segments_from_chunks(chunks, audio_duration, num_images)

    prompts = generate_prompts_from_transcript_segments(segments, audio_part, model, num_images)

    # durations per segment for exact sync
    durations = []
    for seg in segments:
        d = max(0.1, float(seg["end"] - seg["start"]))
        durations.append(d)

    # fix rounding drift so sum(durations) == audio_duration (if known)
    if audio_duration > 0 and durations:
        drift = audio_duration - sum(durations)
        durations[-1] = max(0.1, durations[-1] + drift)

    return transcript, prompts, durations


def _pick_font(base_font_size: int):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, base_font_size), p
            except Exception:
                pass
    return ImageFont.load_default(), None


def add_text_overlay(image_path: str, text: str, output_path: str) -> str:
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # slightly larger baseline for better visibility
    base_font_size = int(height * 0.06)
    if len(text) > 20:
        base_font_size = int(base_font_size * 0.75)
    elif len(text) > 15:
        base_font_size = int(base_font_size * 0.85)

    font, chosen_font_path = _pick_font(base_font_size)

    # clamp overlay length (keeps it readable)
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) > 60:
        text = text[:57].rstrip() + "..."

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    max_text_width = int(width * 0.9)
    if text_width > max_text_width:
        scale_factor = max_text_width / max(text_width, 1)
        new_size = max(12, int(base_font_size * scale_factor))
        if chosen_font_path:
            try:
                font = ImageFont.truetype(chosen_font_path, new_size)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

    # place near bottom center (more “subtitle-like”)
    x = (width - text_width) // 2
    y = int(height * 0.78)

    padding = int(height * 0.02)
    bg_bbox = [x - padding, y - padding, x + text_width + padding, y + text_height + padding]
    draw.rectangle(bg_bbox, fill=(0, 0, 0, 220))

    outline_width = 2
    for ox in range(-outline_width, outline_width + 1):
        for oy in range(-outline_width, outline_width + 1):
            if ox == 0 and oy == 0:
                continue
            draw.text((x + ox, y + oy), text, font=font, fill=(0, 0, 0, 255))

    draw.text((x, y), text, font=font, fill=(255, 215, 0, 255))
    img.save(output_path)
    return output_path


def generate_images(prompts):
    image_files = []
    os.makedirs("generated_frames", exist_ok=True)

    try:
        print("Loading Imagen 4 model...")
        model = ImageGenerationModel.from_pretrained("imagen-4.0-fast-generate-001")
        print("✓ Using Imagen 4 Fast")
    except Exception as e:
        print(f"⚠️  Imagen 4 not available: {e}")
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
        print("⚠️  Using Imagen 3")

    for i, prompt_line in enumerate(prompts):
        print(f"\nProcessing image {i+1}/{len(prompts)}...")

        text_overlay = None
        prompt = prompt_line

        text_match = re.search(r"\[TEXT:\s*([^\]]+)\]", prompt_line)
        if text_match:
            text_overlay = text_match.group(1).strip()
            prompt = re.sub(r"\[TEXT:[^\]]+\]", "", prompt_line).strip()
            print(f"  📝 Text overlay: {text_overlay}")
        else:
            # ensure overlay always exists
            text_overlay = "Key update"

        if i > 0:
            time.sleep(3)

        max_retries = 5
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                images = model.generate_images(
                    prompt=prompt,
                    number_of_images=1,
                    aspect_ratio=ASPECT_RATIO,
                    add_watermark=False,
                )

                temp_filename = f"generated_frames/frame_{i:03d}_temp.png"
                final_filename = f"generated_frames/frame_{i:03d}.png"
                images[0].save(location=temp_filename)

                # always overlay text
                add_text_overlay(temp_filename, text_overlay, final_filename)
                os.remove(temp_filename)

                image_files.append(final_filename)
                print(f"  ✓ Saved: {final_filename}")
                break

            except Exception as e:
                msg = str(e)
                if ("429" in msg or "Quota exceeded" in msg) and attempt < max_retries - 1:
                    print(f"  ⚠️  Rate limit hit; retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"  ✗ Image failed: {e}")
                    if image_files:
                        image_files.append(image_files[-1])
                        print(f"  → Using fallback: {image_files[-1]}")
                    break

    return image_files


def create_zooming_clip(image_path, duration, zoom_ratio=1.3):
    clip = ImageClip(image_path).with_duration(duration)
    w, h = clip.size

    def zoom_out_effect(get_frame, t):
        frame = get_frame(t)
        progress = t / duration if duration else 1.0
        current_zoom = zoom_ratio - (zoom_ratio - 1.0) * progress

        current_w = int(w * current_zoom)
        current_h = int(h * current_zoom)

        img = Image.fromarray(frame)
        img_resized = img.resize((current_w, current_h), Image.LANCZOS)

        left = (current_w - w) // 2
        top = (current_h - h) // 2
        img_cropped = img_resized.crop((left, top, left + w, top + h))
        return np.array(img_cropped)

    return clip.transform(zoom_out_effect)


def create_video(audio_path, image_files, output_path, durations_per_image):
    print("\nStitching video together with zoom effects...")

    # use per-image durations (sync fix)
    clips = [
        create_zooming_clip(img, float(d), zoom_ratio=1.3)
        for img, d in zip(image_files, durations_per_image)
    ]

    video = concatenate_videoclips(clips, method="compose")

    audio = AudioFileClip(audio_path)
    final_video = video.with_audio(audio)

    print("\nRendering final video...")
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")

    audio.close()
    final_video.close()
    print(f"✅ Created: {output_path}")


# -----------------------
# SEO METADATA JSON
# -----------------------
def generate_youtube_metadata_json(transcript: str, channel_name: str) -> dict:
    model = GenerativeModel("gemini-2.0-flash-exp")

    prompt = f"""
You are an expert YouTube SEO copywriter.

Channel: {channel_name}
Content type: YouTube Shorts
Topic: finance / investing / Tech/ AI / USMarket/ money (use transcript to infer exact topic)

Transcript:
{transcript}

Create SEO-optimized metadata in STRICT JSON ONLY (no markdown, no commentary), with exactly these keys:
{{
  "title": "...",
  "description": "...",
  "hashtags": ["#tag1", "#tag2", "..."]
}}

Rules:
- Title must be <= 100 characters.
- Description must be <= 500 characters.
- Put the primary keyword in the first 50 characters of the title.
- First 150 characters of description should be a strong hook + topic summary.
- Include a short CTA to subscribe to "{channel_name}" near the top.
- Hashtags: 5 to 12 total, geopolitics-relevant, include #shorts, and avoid duplicates.
- Hashtags must start with # and contain no spaces.
"""

    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()

    try:
        data = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("Model did not return JSON.")
        data = json.loads(text[start: end + 1])

    data["hashtags"] = [h.strip() for h in data.get("hashtags", []) if isinstance(h, str) and h.strip().startswith("#")]
    data["title"] = str(data.get("title", "")).strip()
    data["description"] = str(data.get("description", "")).strip()
    return data


def save_metadata_json(metadata: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved metadata JSON: {path}")


def main():
    print("=" * 60)
    print("YouTube Shorts Video Generator using Vertex AI")
    print("=" * 60)

    init_vertex_ai()

    if not MP3_FILE:
        print("❌ Error: MP3_FILE env var not set.")
        return
    if not os.path.exists(MP3_FILE):
        print(f"❌ Error: Audio file not found: {MP3_FILE}")
        return

    duration = get_audio_duration(MP3_FILE)
    if duration <= 0:
        return

    num_images = math.ceil(duration / IMAGE_DURATION)
    print(f"Audio duration: {duration:.2f}s -> generating {num_images} images")

    transcript, prompts, durations_per_image = generate_prompts_and_transcript_from_audio(
        MP3_FILE, num_images, duration
    )

    metadata = generate_youtube_metadata_json(transcript, CHANNEL_NAME)
    save_metadata_json(metadata, OUTPUT_METADATA)

    print(f"Generated {len(prompts)} prompts")
    image_paths = generate_images(prompts)

    if image_paths:
        # ensure durations list matches images list
        if len(durations_per_image) != len(image_paths):
            durations_per_image = [duration / max(1, len(image_paths))] * len(image_paths)
        create_video(MP3_FILE, image_paths, OUTPUT_VIDEO, durations_per_image)
    else:
        print("❌ No images were generated. Cannot create video.")


if __name__ == "__main__":
    main()
