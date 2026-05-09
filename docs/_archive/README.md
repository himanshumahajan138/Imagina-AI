# Archive

Historical / pre-implementation docs that are no longer accurate but worth keeping for context.

When a doc here is referenced from a live doc (e.g. an architecture-decision rationale), the live doc should make clear it's archival — readers shouldn't have to discover that themselves.

## What goes here

- **Pre-implementation plans** that have since been built. The original `worker-service.md` plan was an example — once the worker existed, the plan was superseded by [../worker.md](../worker.md). Keep the plan when its rationale and trade-off discussion is useful for understanding *why* the as-built shape exists.
- **Deprecated subsystem docs** for code that's been removed but had non-trivial design discussion.
- **Obsolete approaches** that we tried, ruled out, and want to avoid re-litigating.

## What does NOT go here

- Stale-but-still-correct docs — fix them in place instead.
- Notes that haven't graduated to a real doc — those belong in your scratch dir, not the repo.
- Issue trackers / TODOs — those go in [../roadmap.md](../roadmap.md) or GitHub issues.

Currently empty. Drop the `worker-service.md` plan here if you find it kicking around.
