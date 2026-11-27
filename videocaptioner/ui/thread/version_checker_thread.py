# coding: utf-8
"""Version checker for the Intel compatibility fork.

This branch publishes its own installers on GitHub Releases with tags such as
``v1.4.3-intel.10``. The upstream JSON API reports upstream versions, which do
not apply here, so updates are resolved from this fork's release list and
compared by the ``-intel.N`` build number.
"""

import hashlib
import os
import re
from datetime import datetime

import requests
from PyQt5.QtCore import QObject, pyqtSignal

from videocaptioner.config import (
    BRANCH_NAME,
    INTEL_BUILD,
    RELEASES_API,
    VERSION,
)
from videocaptioner.core.utils.cache import get_version_state_cache
from videocaptioner.core.utils.logger import setup_logger

logger = setup_logger("version_checker")

_TAG_RE = re.compile(r"^v?[\d.]+-intel\.(\d+)$")


class VersionChecker(QObject):
    """Version checker"""

    newVersionAvailable = pyqtSignal(str, bool, str, str)
    announcementAvailable = pyqtSignal(str)
    checkCompleted = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_version = VERSION
        self.latest_version = VERSION
        self.latest_build = INTEL_BUILD
        self.update_info = ""
        self.update_required = False
        self.download_url = ""
        self.announcement = {}

        self.cache = get_version_state_cache()

    @staticmethod
    def _headers() -> dict:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": f"VideoCaptioner-IntelArc/{VERSION}",
        }
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _fetch_intel_releases(self) -> list[tuple[int, dict]]:
        """Return Intel releases as ``(build_number, release)`` sorted desc."""
        response = requests.get(RELEASES_API, headers=self._headers(), timeout=10)
        response.raise_for_status()
        releases: list[tuple[int, dict]] = []
        for release in response.json():
            if release.get("draft") or release.get("prerelease"):
                continue
            match = _TAG_RE.match(str(release.get("tag_name", "")))
            if match:
                releases.append((int(match.group(1)), release))
        releases.sort(key=lambda item: item[0], reverse=True)
        return releases

    def get_latest_version_info(self) -> dict:
        """Get latest Intel build information from this fork's releases."""
        try:
            releases = self._fetch_intel_releases()
        except requests.RequestException as exc:
            logger.debug("Failed to query %s releases: %s", BRANCH_NAME, exc)
            return {}
        except Exception:
            logger.exception("Unexpected error while querying releases")
            return {}

        if not releases:
            logger.info("No Intel release found for %s", BRANCH_NAME)
            return {}

        build, release = releases[0]
        self.latest_build = build
        self.latest_version = str(release.get("tag_name", f"v{VERSION}-intel.{build}"))
        self.update_info = str(release.get("body") or "").strip()[:4000]
        self.download_url = str(
            release.get("html_url") or ""
        ) or "https://github.com/liu-nexotic/VideoCaptioner/releases"
        self.update_required = False
        self.announcement = {}

        logger.info(
            "Latest Intel build: %s (current build %s)", self.latest_version, INTEL_BUILD
        )
        return {
            "latest_version": self.latest_version,
            "latest_build": self.latest_build,
            "update_required": self.update_required,
            "update_info": self.update_info,
            "download_url": self.download_url,
            "announcement": self.announcement,
        }

    def has_new_version(self) -> bool:
        """Check whether a newer Intel build has been published."""
        if self.latest_build <= INTEL_BUILD:
            return False

        logger.info(
            "New Intel build found: %s (current build %s)",
            self.latest_version,
            INTEL_BUILD,
        )
        self.newVersionAvailable.emit(
            self.latest_version,
            self.update_required,
            self.update_info,
            self.download_url,
        )
        return True

    def check_announcement(self) -> None:
        """Check and show announcement"""
        ann = self.announcement
        if not ann.get("enabled", False):
            return

        content = ann.get("content", "")
        if not content:
            return

        announcement_id = (
            hashlib.sha256(content.encode("utf-8")).hexdigest()
            + "_"
            + datetime.today().strftime("%Y-%m-%d")
        )

        settings_key = f"announcement/shown_{announcement_id}"
        if self.cache.get(settings_key, default=False):
            return

        start_date_str = ann.get("start_date")
        end_date_str = ann.get("end_date")
        if not start_date_str or not end_date_str:
            return

        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            today = datetime.today().date()

            if start_date <= today <= end_date:
                self.cache.set(settings_key, True, expire=30 * 60 * 24)
                self.announcementAvailable.emit(content)

        except ValueError as e:
            logger.error("Announcement date format error: %s", str(e))

    def check_new_version_announcement(self) -> None:
        """Previous-release notes are shown once per build."""
        if self.latest_build != INTEL_BUILD:
            return

        version_key = f"intel-build/shown_{self.latest_version}"

        if not self.cache.get(version_key, default=False):
            self.cache.set(version_key, True)

            update_announcement = (
                f"VideoCaptioner Intel 兼容分支 build {INTEL_BUILD}\n"
                f"分支：{BRANCH_NAME}\n\n"
                f"{self.update_info}"
            )
            self.announcementAvailable.emit(update_announcement)

    def perform_check(self) -> None:
        """Perform version and announcement check"""
        try:
            version_data = self.get_latest_version_info()
            if not version_data:
                return
            self.has_new_version()
            self.check_new_version_announcement()
            self.check_announcement()
            self.checkCompleted.emit()
        except Exception:
            logger.exception("Version and announcement check failed")
