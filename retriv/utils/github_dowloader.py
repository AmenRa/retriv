"""Code based on https://github.com/sdushantha/gitdir"""
import json
import re
import sys
import urllib.request
from pathlib import Path


def home_path():
    p = Path(Path.home() / ".retriv" / "hunspell")
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_url(url):
    """
    From the given url, produce a URL that is compatible with Github's REST API. Can handle blob or tree paths.
    """
    re_branch = re.compile("/(tree|blob)/(.+?)/")

    # extract the branch name from the given url (e.g master)
    branch = re_branch.search(url)
    download_dir = url[branch.end() :]
    api_url = (
        url[: branch.start()].replace("github.com", "api.github.com/repos", 1)
        + "/contents/"
        + download_dir
        + "?ref="
        + branch[2]
    )

    return api_url, download_dir


def download(repo_url):
    """Downloads the files and directories in repo_url. If flatten is specified, the contents of any and all sub-directories will be pulled upwards into the root folder."""

    api_url, download_dir = create_url(repo_url)
    download_dir = home_path() / download_dir

    if download_dir.exists():
        return str(download_dir)

    download_dir.mkdir(parents=True, exist_ok=True)

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    response = urllib.request.urlretrieve(api_url)

    with open(response[0], "r") as f:
        data = json.load(f)

        for file in data:
            file_url = file["download_url"]
            file_path = Path(file["path"]).name

            opener = urllib.request.build_opener()
            opener.addheaders = [("User-agent", "Mozilla/5.0")]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(file_url, download_dir / file_path)

    return str(download_dir)
