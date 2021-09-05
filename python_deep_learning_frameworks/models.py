import math
from datetime import datetime, timedelta

DATETIME_FMT = "%a, %d %b %Y %H:%M:%S %Z"


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
        last_modified = datetime.strptime(self.repo.last_modified, DATETIME_FMT)
        delta = datetime.today() - last_modified
        last_modified_in_days = delta / timedelta(days=1)

        return math.floor(last_modified_in_days)
