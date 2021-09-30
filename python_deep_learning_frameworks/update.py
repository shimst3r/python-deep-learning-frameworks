import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from github import Github
from jinja2 import Template

from models import Repository, DATETIME_FMT


def main():
    access_token = os.environ["GITHUB_TOKEN"]
    g = Github(access_token)
    project_root = Path(__file__).parent

    repos = _get_repositories(project_root, g)
    repos = sorted(repos, key=lambda repo: repo.stargazers_count, reverse=True)
    template = _populate_template(repos)
    _init_csv(project_root)
    _export_csv(project_root, repos)
    _update_readme(template)


def _export_csv(project_root, repositories: List[Repository]):
    with (project_root / "repositories.csv").open("a", encoding="utf8") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        for repo in repositories:
            writer.writerow(repo.to_list())


def _get_repositories(project_root, g):
    with (project_root / "list.txt").open("r", encoding="utf8") as in_file:
        repo_identifiers = [line.strip() for line in in_file.readlines()]
    repositories = [
        Repository(repo=g.get_repo(identifier)) for identifier in repo_identifiers
    ]
    return repositories


def _init_csv(project_root):
    csv_path = project_root / "repositories.csv"

    if not csv_path.exists():
        with csv_path.open("w", encoding="utf8") as csv_file:
            writer = csv.writer(csv_file, delimiter="\t")
            writer.writerow(["timestamp", "name", "stargazers", "forks", "open_issues"])


def _populate_template(project_root, repositories):
    with (project_root / "template.md.j2").open("r", encoding="utf8") as template_file:
        template = Template(template_file.read())
    last_update = datetime.now().astimezone().strftime(DATETIME_FMT)
    return template.render(repositories=repositories, last_update=last_update)


def _update_readme(project_root, template):
    with (project_root / "../README.md").open("w", encoding="utf8") as readme_file:
        readme_file.write(template)


if __name__ == "__main__":
    sys.exit(main())
