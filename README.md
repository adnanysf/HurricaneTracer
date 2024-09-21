# Group Project Repository

This README provides guidelines for collaborating on this project using Git and GitHub.

## Branching Strategy

Each team member should work on their own branch. Follow these steps to create and use your branch:

1. Clone the repository:
`git clone [repository-url]`

2. Create your branch:
`git checkout -b [your-first-name]`

3. Push your branch to GitHub:
`git push -u origin [your-first-name]`

4. Make your changes, commit, and push regularly:
   - `git add .`
   - `git commit -m "message"`
   - `git push`

## Merging Changes

When you're ready to merge your changes into the main branch:

1. Ensure your branch is up to date with the main branch:
   - `git checkout main`
   - `git pull`
   - `git checkout [your-name]`
   - `git merge main`

2. Resolve any conflicts if necessary.

3. Push your updated branch to GitHub:
`git push`

4. Go to the GitHub repository page and create a new Pull Request (PR) from your branch to the main branch.

5. Add a clear title and description to your PR, explaining the changes you've made.

6. Request a review from at least one other team member.

7. Once approved, the PR can be merged into the main branch.
