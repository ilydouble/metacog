# Repository Development Guidelines

This repository should be developed with clear domain boundaries and verified changes.


## Testing And Git

- Every code or documentation change must be committed to git immediately after the change is completed.
- Keep commits scoped to the files changed for the current task.
- Do not bundle unrelated workspace changes into the same commit.
- Every code change must include or update focused tests.
- Run the relevant test slice before broad verification.
- Before handing off or committing, run the full applicable test suite and ensure it passes.
- Commit only after tests pass. Keep git commits scoped to the completed change.