import os
import sys
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaFileUpload

scopes = ["https://www.googleapis.com/auth/youtube.upload"]

def main():
    # Define the scopes
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]

    # Load credentials from client_secrets.json
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        "client_secret.json", scopes=scopes
    )

    # Start local server for authentication
    credentials = flow.run_console()

    # Build the YouTube service
    youtube = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)

    # Upload code continues below...


if __name__ == "__main__":
    main()
