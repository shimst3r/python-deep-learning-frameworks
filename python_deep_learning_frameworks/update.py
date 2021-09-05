import os
import sys
from pathlib import Path
from datetime import datetime

from github import Github
from jinja2 import Template

from models import Repository, DATETIME_FMT

PROJECT_ROOT = Path(__file__).parent


def main():
    access_token = os.environ["GITHUB_TOKEN"]
    g = Github(access_token)

    repositories = sorted(
        _get_repositories(g), key=lambda repo: repo.stargazers_count, reverse=True
    )
    template = _populate_template(repositories)
    _update_readme(template)


def _get_repositories(g):
    with open(PROJECT_ROOT / "list.txt", "r", encoding="utf8") as in_file:
        repo_identifiers = [line.strip() for line in in_file.readlines()]
    repositories = [
        Repository(repo=g.get_repo(identifier)) for identifier in repo_identifiers
    ]
    return repositories


def _populate_template(repositories):
    with open(PROJECT_ROOT / "template.md.j2", "r", encoding="utf8") as template_file:
        template = Template(template_file.read())
    last_update = datetime.now().astimezone().strftime(DATETIME_FMT)
    return template.render(repositories=repositories, last_update=last_update)


def _update_readme(template):
    with open(PROJECT_ROOT / "../README.md", "w", encoding="utf8") as readme_file:
        readme_file.write(template)


if __name__ == "__main__":
    sys.exit(main())
