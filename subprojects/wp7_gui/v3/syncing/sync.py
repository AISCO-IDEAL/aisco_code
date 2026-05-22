from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local

import httplib2
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


class DriveSync:
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    MAX_WORKERS = 4
    SMALL_UPLOAD_LIMIT = 5 * 1024 * 1024
    GOOGLE_APPS_MIME = "application/vnd.google-apps."
    DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"

    def __init__(
        self,
        local_folder=None,
        folder_id="0AOQFbhSrPHtWUk9PVA",
        credentials_file=None,
        token_file=None,
    ):
        base = Path(__file__).resolve().parent
        self.local_folder = Path(local_folder or base / "ROLLOUT_data")
        self.folder_id = folder_id
        self.credentials_file = Path(credentials_file or base / "credentials.json")
        self.token_file = Path(token_file or base / "token.json")
        self.local_folder.mkdir(parents=True, exist_ok=True)
        self.creds = self.auth()
        self.service = build("drive", "v3", credentials=self.creds)
        self.thread = local()

    def auth(self):
        creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES) if self.token_file.exists() else None
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds = InstalledAppFlow.from_client_secrets_file(self.credentials_file, self.SCOPES).run_local_server(port=0)
        self.token_file.write_text(creds.to_json(), encoding="utf-8")
        return creds

    def children(self, parent_id):
        page_token = None
        while True:
            result = (
                self.service.files()
                .list(
                    q=f"'{parent_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType)",
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                    pageSize=100,
                    pageToken=page_token,
                )
                .execute()
            )
            yield from result.get("files", [])
            page_token = result.get("nextPageToken")
            if not page_token:
                return

    def create_folder(self, name, parent_id):
        return (
            self.service.files()
            .create(
                body={"name": name, "mimeType": self.DRIVE_FOLDER_MIME, "parents": [parent_id]},
                fields="id, name, mimeType",
                supportsAllDrives=True,
            )
            .execute()
        )

    def worker(self):
        if not hasattr(self.thread, "drive"):
            http = AuthorizedHttp(self.creds, http=httplib2.Http())
            self.thread.drive = build("drive", "v3", http=http, cache_discovery=False)
        return self.thread.drive

    def download(self, file, target_path):
        request = self.worker().files().get_media(fileId=file["id"], supportsAllDrives=True)
        with target_path.open("wb") as target:
            downloader, done = MediaIoBaseDownload(target, request), False
            while not done:
                _, done = downloader.next_chunk()
        return f"Downloaded: {target_path}"

    def upload(self, path, parent_id):
        media = MediaFileUpload(str(path), resumable=path.stat().st_size > self.SMALL_UPLOAD_LIMIT)
        self.worker().files().create(
            body={"name": path.name, "parents": [parent_id]},
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
        return f"Uploaded: {path}"

    def collect_transfers(self, local_folder, parent_id, uploads, downloads):
        files, folders, skipped = {}, {}, set()
        for path in local_folder.iterdir():
            if path.is_symlink() and path.is_dir():
                print(f"Skipped symlinked local folder: {path}")
                skipped.add(path.name)
            elif path.is_dir():
                folders[path.name] = path
            elif path.is_file():
                files[path.name] = path

        drive, duplicates = {}, set()
        for child in self.children(parent_id):
            name = child["name"]
            if name in drive:
                duplicates.add(name)
            drive[name] = child
        for name in duplicates:
            drive.pop(name)
            skipped.add(name)
            print(f"Skipped ambiguous Drive path under {local_folder}: duplicate child name {name!r}")

        for name in sorted(set(files) | set(folders) | set(drive)):
            if name in skipped:
                continue

            local_file, local_folder_child, item = files.get(name), folders.get(name), drive.get(name)
            if item is None:
                if local_file:
                    uploads.append((local_file, parent_id))
                else:
                    item = self.create_folder(name, parent_id)
                    self.collect_transfers(local_folder_child, item["id"], uploads, downloads)
                continue

            if item["mimeType"] == self.DRIVE_FOLDER_MIME:
                if local_file:
                    print(f"Skipped path conflict between a file and a folder: {local_file}")
                else:
                    local_folder_child = local_folder_child or local_folder / name
                    local_folder_child.mkdir(parents=True, exist_ok=True)
                    self.collect_transfers(local_folder_child, item["id"], uploads, downloads)
                continue

            if local_folder_child:
                print(f"Skipped path conflict between a file and a folder: {local_folder_child}")
            elif item["mimeType"].startswith(self.GOOGLE_APPS_MIME):
                print(f"Skipped unsupported Google file: {local_folder / name}")
            elif local_file is None:
                downloads.append((item, local_folder / name))

    def sync(self):
        uploads, downloads = [], []
        self.collect_transfers(self.local_folder, self.folder_id, uploads, downloads)
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            tasks = [pool.submit(self.download, *job) for job in downloads]
            tasks += [pool.submit(self.upload, *job) for job in uploads]
            for task in as_completed(tasks):
                print(task.result())


if __name__ == "__main__":
    DriveSync().sync()
