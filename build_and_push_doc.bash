#!/bin/bash
# This script is used to compile the tutorials of the project.

# Current branch
branch=$(git rev-parse --abbrev-ref HEAD)

Assert that the current branch is the main branch
if [ "$branch" != "main" ]; then
    echo "You must be on the main branch to build the documentation."
    exit 1
fi

# Create a new branch for the documentation
git branch -D docs
git checkout -b docs

bash build_doc.bash

# Add the documentation to the repository
git add docs/ -f
# Commit the changes and push them, without the pre-commit hook that clean up the notebooks
git commit -m "Build documentation"
git push public docs --force

# Back to the current branch
git checkout $branch