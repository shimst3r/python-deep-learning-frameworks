import os

from github import Github

if __name__ == '__main__':
    access_token = os.environ['GITHUB_TOKEN']
    g = Github(access_token)
    repo = g.get_repo('shimst3r/python-deep-learning-frameworks')
    print(repo.name)
