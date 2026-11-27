"""Version checker tests for the Intel fork update source (offline)."""

import videocaptioner.config as config
from videocaptioner.ui.thread.version_checker_thread import VersionChecker


def _release(build: int, draft: bool = False, prerelease: bool = False):
    return (
        build,
        {
            "tag_name": f"v1.4.3-intel.{build}",
            "body": "notes",
            "html_url": f"https://example.invalid/releases/{build}",
            "draft": draft,
            "prerelease": prerelease,
        },
    )


def test_update_source_points_to_intel_fork():
    assert config.BRANCH_NAME == "feat/intel-arc-openvino"
    assert "liu-nexotic/VideoCaptioner" in config.GITHUB_REPO_URL
    assert config.BRANCH_NAME in config.HELP_URL
    assert config.RELEASE_URL.endswith("/releases")
    assert config.FEEDBACK_URL.endswith("/issues")
    assert config.RELEASES_API.endswith("/releases?per_page=20")


def test_no_update_when_latest_build_is_current(monkeypatch):
    checker = VersionChecker()
    emitted = []
    checker.newVersionAvailable.connect(
        lambda version, required, info, url: emitted.append(version)
    )
    monkeypatch.setattr(
        checker, "_fetch_intel_releases", lambda: [_release(config.INTEL_BUILD)]
    )
    checker.get_latest_version_info()
    assert checker.has_new_version() is False
    assert emitted == []


def test_newer_build_emits_update(monkeypatch):
    checker = VersionChecker()
    emitted = []
    checker.newVersionAvailable.connect(
        lambda version, required, info, url: emitted.append((version, url, info))
    )
    newer = config.INTEL_BUILD + 1
    monkeypatch.setattr(checker, "_fetch_intel_releases", lambda: [_release(newer)])
    checker.get_latest_version_info()
    assert checker.has_new_version() is True
    assert emitted
    assert emitted[0][0] == f"v1.4.3-intel.{newer}"
    assert emitted[0][1].endswith(f"/{newer}")
    assert emitted[0][2] == "notes"


def test_older_build_does_not_emit(monkeypatch):
    checker = VersionChecker()
    emitted = []
    checker.newVersionAvailable.connect(lambda *args: emitted.append(args))
    monkeypatch.setattr(
        checker, "_fetch_intel_releases", lambda: [_release(config.INTEL_BUILD - 1)]
    )
    checker.get_latest_version_info()
    assert checker.has_new_version() is False
    assert emitted == []


def test_release_tag_parsing_filters_non_intel_and_prereleases(monkeypatch):
    checker = VersionChecker()

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"tag_name": "v1.4.3-intel.3", "body": "", "draft": False, "prerelease": False},
                {"tag_name": "v1.4.3-intel.9", "body": "", "draft": False, "prerelease": True},
                {"tag_name": "v1.4.3", "body": "", "draft": False, "prerelease": False},
                {"tag_name": "v2.0.0-intel.12", "body": "", "draft": False, "prerelease": False},
            ]

    monkeypatch.setattr(
        "videocaptioner.ui.thread.version_checker_thread.requests.get",
        lambda *a, **k: FakeResponse(),
    )
    releases = checker._fetch_intel_releases()
    assert [build for build, _ in releases] == [12, 3]
