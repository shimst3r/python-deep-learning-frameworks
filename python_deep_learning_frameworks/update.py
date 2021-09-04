import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from github import Github
from jinja2 import Template

PROJECT_ROOT = Path(__file__).parent
DATETIME_IN = "%a, %d %b %Y %H:%M:%S %Z"


class Repository:
    def __init__(self, repo):
        self.repo = repo

    @property
    def name(self):
        return self.repo.name

    @property
    def organization_name(self):
        return self.repo.organization.name

    @property
    def description(self):
        return self.repo.description

    @property
    def stargazers_count(self):
        return self.repo.stargazers_count

    @property
    def forks_count(self):
        return self.repo.forks_count

    @property
    def open_issues_count(self):
        return self.repo.open_issues_count

    @property
    def last_commit(self):
        last_modified = datetime.strptime(self.repo.last_modified, DATETIME_IN)
        delta = datetime.today() - last_modified
        last_modified_in_days = delta / timedelta(days=1)

        return math.floor(last_modified_in_days)


def get_repositories(g):
    with open(PROJECT_ROOT / "list.txt", "r", encoding="utf8") as in_file:
        repo_identifiers = [line.strip() for line in in_file.readlines()]
    repositories = [
        Repository(repo=g.get_repo(identifier)) for identifier in repo_identifiers
    ]
    return repositories


def populate_template(repositories):
    with open(PROJECT_ROOT / "template.md.j2", "r", encoding="utf8") as template_file:
        template = Template(template_file.read())
    return template.render(repositories=repositories)

def update_readme(template):
    with open(PROJECT_ROOT / '../README.md', 'w', encoding='utf8') as readme_file:
        readme_file.write(template)

def main():
    access_token = os.environ["GITHUB_TOKEN"]
    g = Github(access_token)

    repositories = sorted(
        get_repositories(g), key=lambda repo: repo.stargazers_count, reverse=True
    )
    template = populate_template(repositories)
    update_readme(template)


if __name__ == "__main__":
    sys.exit(main())
