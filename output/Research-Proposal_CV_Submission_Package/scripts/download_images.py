#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vlm_gap.config import DEFAULT_IMAGE_DIR, DEFAULT_MANIFEST  # noqa: E402
from vlm_gap.data import load_manifest  # noqa: E402


ARCHIVE_URL = (
    "https://huggingface.co/datasets/nlphuji/dollar_street_test/"
    "resolve/main/dollarstreet_test_images.zip"
)


class HTTPRangeReader(io.RawIOBase):
    """Seekable HTTP reader that downloads only requested byte ranges."""

    def __init__(self, url: str, timeout: int = 90):
        self.timeout = timeout
        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "vlm-income-gap-course-project/0.1"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            self.url = response.geturl()
            self.length = int(response.headers["Content-Length"])
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"Unknown seek mode: {whence}")
        if position < 0:
            raise ValueError("Cannot seek before the start of the archive")
        self.position = position
        return self.position

    def read(self, size=-1):
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            end = self.length - 1
        else:
            end = min(self.length - 1, self.position + size - 1)
        if end < self.position:
            return b""

        content = None
        last_error = None
        for attempt in range(4):
            request = urllib.request.Request(
                self.url,
                headers={
                    "Range": f"bytes={self.position}-{end}",
                    "User-Agent": "vlm-income-gap-course-project/0.1",
                },
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout,
                ) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            "The archive server did not honor the byte-range "
                            "request; aborting to avoid downloading the full "
                            "11 GB archive."
                        )
                    content = response.read()
                break
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2 ** attempt)
        if content is None:
            raise RuntimeError(
                f"Byte-range download failed after four attempts: {last_error}"
            ) from last_error
        self.position += len(content)
        return content

    def readinto(self, buffer):
        content = self.read(len(buffer))
        buffer[: len(content)] = content
        return len(content)


def valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    args.image_dir.mkdir(parents=True, exist_ok=True)

    remote_archive = io.BufferedReader(
        HTTPRangeReader(ARCHIVE_URL),
        buffer_size=4 * 1024 * 1024,
    )
    with zipfile.ZipFile(remote_archive) as archive:
        archive_names = set(archive.namelist())
        for index, row in enumerate(rows, start=1):
            destination = row.image_path(args.image_dir)
            if valid_image(destination):
                continue
            print(f"[{index}/{len(rows)}] Downloading {row.image_name}")
            member = f"dollarstreet_test_images/{row.image_name}"
            if member not in archive_names:
                raise KeyError(f"Image is missing from the source archive: {member}")

            partial = destination.with_suffix(destination.suffix + ".part")
            with archive.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            partial.replace(destination)
            if not valid_image(destination):
                destination.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Downloaded file is not a valid image: {destination}"
                )

    print(f"Images available in {args.image_dir}")


if __name__ == "__main__":
    main()
