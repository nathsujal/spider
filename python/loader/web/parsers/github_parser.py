import re
import httpx
from urllib.parse import urlparse
from pathlib import Path

from ..utils import get


class GithubHandler:
    """
    Resolves GitHub URLs and fetches raw text content.

    URL shapes handled:
      /user/repo                       → fetch README.md
      /user/repo/blob/BRANCH/path/file → fetch raw file content
      /user/repo/tree/BRANCH/folder    → fetch folder README if present

    Once raw content is obtained, delegates to TextLoader's parsers
    using the file extension — so a .py file gets PlainParser,
    a .md file gets MarkdownParser, etc.
    """

    # Matches: /user/repo/blob/branch/path/to/file
    _BLOB_RE = re.compile(
        r"^/(?P<user>[^/]+)/(?P<repo>[^/]+)/blob/(?P<branch>[^/]+)/(?P<path>.+)$"
    )
    # Matches: /user/repo/tree/branch/folder
    _TREE_RE = re.compile(
        r"^/(?P<user>[^/]+)/(?P<repo>[^/]+)/tree/(?P<branch>[^/]+)/(?P<folder>.+)$"
    )
    # Matches: /user/repo  (repo root, optional trailing slash)
    _REPO_RE = re.compile(
        r"^/(?P<user>[^/]+)/(?P<repo>[^/]+)/?$"
    )

    _RAW_BASE = "https://raw.githubusercontent.com"
    _README_CANDIDATES = ["README.md", "README.rst", "README.txt", "README"]

    def fetch(self, url: str) -> tuple[str, str, str]:
        """
        Returns (raw_content, file_extension, resolved_url).
        Raises ValueError if no readable content can be found.
        """
        parsed = urlparse(url)
        path   = parsed.path

        if m := self._BLOB_RE.match(path):
            return self._fetch_raw_file(
                m["user"], m["repo"], m["branch"], m["path"]
            )

        if m := self._TREE_RE.match(path):
            return self._fetch_folder_readme(
                m["user"], m["repo"], m["branch"], m["folder"]
            )

        if m := self._REPO_RE.match(path):
            return self._fetch_repo_readme(m["user"], m["repo"])

        raise ValueError(f"Unrecognised GitHub URL shape: {url}")

    def _fetch_raw_file(
        self, user: str, repo: str, branch: str, file_path: str
    ) -> tuple[str, str, str]:
        raw_url = f"{self._RAW_BASE}/{user}/{repo}/{branch}/{file_path}"
        resp = get(raw_url)
        ext = Path(file_path).suffix.lower() or ".txt"
        return resp.text, ext, raw_url

    def _fetch_folder_readme(
        self, user: str, repo: str, branch: str, folder: str
    ) -> tuple[str, str, str]:
        for name in self._README_CANDIDATES:
            raw_url = f"{self._RAW_BASE}/{user}/{repo}/{branch}/{folder}/{name}"
            try:
                resp = get(raw_url)
                ext = Path(name).suffix.lower() or ".txt"
                return resp.text, ext, raw_url
            except httpx.HTTPStatusError:
                continue

        raise ValueError(
            f"No README found in {user}/{repo}/{folder}. "
            "Try linking directly to a file instead."
        )

    def _fetch_repo_readme(self, user: str, repo: str) -> tuple[str, str, str]:
        # Try default branches: main → master → HEAD
        for branch in ("main", "master", "HEAD"):
            for name in self._README_CANDIDATES:
                raw_url = f"{self._RAW_BASE}/{user}/{repo}/{branch}/{name}"
                try:
                    resp = get(raw_url)
                    ext = Path(name).suffix.lower() or ".txt"
                    return resp.text, ext, raw_url
                except httpx.HTTPStatusError:
                    continue

        raise ValueError(
            f"Could not find a README in {user}/{repo}. "
            "Try linking directly to a file."
        )