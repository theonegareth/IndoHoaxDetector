# 📤 **Guide for Uploading Large Model Files**

Since GitHub has file size limits (~100MB) and our IndoBERT model files are large, here are several approaches to share the trained models:

## 🚀 **Option 1: Git LFS (Recommended for GitHub)**

### **Setup Git LFS:**
```bash
# Install Git LFS (if not already installed)
git lfs install

# Track large files
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "*.pth"
git lfs track "*.pt"

# Add the tracking file
git add .gitattributes

# Add large files with LFS
git add indobert_model/model.safetensors
git add indobert_experiments/*/checkpoint-*/model.safetensors
git add indobert_experiments/*/checkpoint-*/*.bin
git add indobert_experiments/*/checkpoint-*/*.pth
git add indobert_experiments/*/checkpoint-*/*.pt

# Commit and push
git commit -m "feat: Add IndoBERT model files via Git LFS"
git push origin main
```

## 📁 **Option 2: Cloud Storage Upload**

### **Google Drive Upload:**
```bash
# Install gdrive CLI tool
# Download from: https://github.com/prasmussen/gdrive

# Upload main model
gdrive upload indobert_model/model.safetensors

# Upload experiment checkpoints
for dir in indobert_experiments/*/checkpoint-*/; do
    gdrive upload "$dir"model.safetensors
done
```

### **Dropbox Upload:**
```bash
# Use Dropbox CLI or web interface
# Upload to: /IndoHoaxDetector/models/
```

### **AWS S3 Upload:**
```bash
# Configure AWS CLI
aws configure

# Create S3 bucket
aws s3 mb s3://indohoaxdetector-models

# Upload models
aws s3 cp indobert_model/model.safetensors s3://indohoaxdetector-models/
aws s3 sync indobert_experiments/ s3://indohoaxdetector-models/experiments/ --exclude "*" --include "*/model.safetensors"
```

## 🐍 **Option 3: Python Upload Script**

<write_to_file>
<path>upload_models_to_drive.py</path>
<content>"""
Upload large model files to Google Drive using Google Drive API
"""

import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_google_drive():
    """Authenticate and return Google Drive service."""
    creds = None
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)

def upload_file(service, file_path, folder_id=None):
    """Upload a file to Google Drive."""
    file_name = os.path.basename(file_path)
    
    file_metadata = {'name': file_name}
    if folder_id:
        file_metadata['parents'] = [folder_id]
    
    media = MediaFileUpload(file_path, resumable=True)
    
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    
    print(f"Uploaded {file_name} with ID: {file.get('id')}")
    return file.get('id')

def main():
    """Main function to upload model files."""
    service = authenticate_google_drive()
    
    # Create folder for models
    folder_metadata = {
        'name': 'IndoHoaxDetector_Models',
        'mimeType': 'application/vnd.google-apps.folder'
    }
    
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    folder_id = folder.get('id')
    print(f"Created folder with ID: {folder_id}")
    
    # Upload main model
    main_model_path = 'indobert_model/model.safetensors'
    if os.path.exists(main_model_path):
        upload_file(service, main_model_path, folder_id)
    
    # Upload experiment models
    experiments_dir = 'indobert_experiments'
    if os.path.exists(experiments_dir):
        for root, dirs, files in os.walk(experiments_dir):
            for file in files:
                if file == 'model.safetensors':
                    file_path = os.path.join(root, file)
                    upload_file(service, file_path, folder_id)
    
    print("Upload completed!")

if __name__ == "__main__":
    main()