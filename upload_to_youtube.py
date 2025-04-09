import os
import sys
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaFileUpload

scopes = ["https://www.googleapis.com/auth/youtube.upload"]

def main():
    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = os.environ["CLIENT_SECRET_FILE"]

    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, scopes)
    credentials = flow.run_console()

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials)

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": os.environ.get("TITLE", "My Video"),
                "description": os.environ.get("DESCRIPTION", ""),
                "categoryId": os.environ.get("CATEGORY", "22")
            },
            "status": {
                "privacyStatus": os.environ.get("PRIVACY", "unlisted")
            }
        },
        media_body=MediaFileUpload(os.environ["VIDEO_FILE"])
    )
    response = request.execute()
    print("✅ Video uploaded. YouTube video ID:", response.get("id"))

if __name__ == "__main__":
    main()
