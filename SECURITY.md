# Security Policy

Thanks for helping keep Imagina AI and its users safe. This document explains how to report vulnerabilities and what to expect after you do.

## Supported versions

Imagina AI is an actively developed solo project. Security fixes are applied to the `main` branch only. There are no long-lived release branches — please test against the latest commit on `main` before reporting.

| Version | Supported |
| ------- | --------- |
| `main` (latest) | ✅ |
| Older commits / forks | ❌ |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Report privately via either channel:

- **Email:** <himanshumahajan138@gmail.com> — preferred. Use the subject line `[SECURITY] Imagina AI: <short summary>`.
- **GitHub private advisory:** open a draft advisory at <https://github.com/himanshumahajan138/Imagina-AI/security/advisories/new>.

When reporting, please include as much of the following as you can:

- A clear description of the issue and the impact.
- Steps to reproduce — ideally a minimal proof of concept.
- The affected file(s), commit SHA, and the tier/backend involved (Local, Cloud OSS, or API) if relevant.
- Your environment (OS, Python version, provider whose key was in use if applicable).
- Any logs, screenshots, or sample inputs. **Redact API keys, tokens, and personal data before sending.**
- Whether you'd like to be credited in the fix announcement (and the name/handle to use).

## What to expect

- **Acknowledgement:** within 72 hours.
- **Initial triage:** within 7 days — confirmation of the issue, severity assessment, and a rough fix timeline.
- **Fix and disclosure:** depends on severity and complexity. Critical issues are prioritised; lower-severity ones are batched into the next reasonable change.
- **Credit:** with your permission, contributors are named in the fix commit / release notes.

Please give us a reasonable window to ship a fix before publishing details. Coordinated disclosure protects users running deployments.

## Scope

In scope:

- Code in this repository (`core/`, `worker/`, `server/`, `pipelines/`, `services/`, `ui/`, `app.py`).
- Configuration handling — especially `.env` parsing, secret loading, and the model registry.
- The FastAPI worker (`worker/`) and any endpoint it exposes.
- File-handling paths (uploads, ffmpeg invocations, YouTube downloader, watermark/trim tools) — particularly anything touching user-supplied paths, URLs, or media.
- Streamlit UI session handling and any state shared between users in a multi-user deployment.

Out of scope:

- Vulnerabilities in third-party providers (OpenAI, Gemini, Replicate, Sync.so, ElevenLabs, etc.) — report those upstream.
- Vulnerabilities in third-party Python packages — report those to the respective project (you can still let us know so we can pin/upgrade).
- Issues that require a pre-compromised host, an attacker with root/admin on the machine running the app, or physical access.
- Denial of service caused by very large model inputs on resource-constrained hardware (Imagina AI targets M2 / 16 GB; OOMing the worker with a 10 GB prompt is expected behaviour, not a vulnerability).
- Missing security headers or rate limiting on a local-dev Streamlit instance. If you deploy publicly, you are responsible for the reverse proxy / WAF in front of it.
- Social-engineering, phishing, or attacks on the maintainer's accounts.

## Examples of issues we want to hear about

- Command injection via filenames, URLs, or prompts (ffmpeg, yt-dlp, shell invocations).
- Path traversal through user-controlled paths into `output/`, `images/`, or `logs/`.
- SSRF or unsafe URL fetching in the YouTube downloader or any provider client.
- Secret leakage — API keys logged, written to outputs, or exposed via UI/error pages.
- Worker endpoints that can be reached cross-origin or by unauthenticated callers when they shouldn't be.
- Unsafe deserialisation (pickle, custom loaders) in model/checkpoint handling.
- Dependency vulnerabilities that meaningfully impact this project's runtime.

## Safe-harbour

If you make a good-faith effort to comply with this policy when researching and reporting an issue, we will not pursue or support any legal action against you. Please avoid:

- Accessing data that doesn't belong to you.
- Degrading service for other users.
- Running automated scanners against production deployments you don't own.
- Publicly disclosing the issue before a fix is shipped or a coordinated disclosure date is agreed.

## Hardening notes for operators

If you deploy Imagina AI beyond `localhost`:

- Put the Streamlit UI (`:8004`) and worker (`:8005`) behind a reverse proxy with auth.
- Never expose the worker port directly to the internet — it executes model code on user inputs.
- Keep `.env` out of version control and off shared volumes. Rotate provider keys you suspect have leaked.
- Run the worker as an unprivileged user; restrict write access to `output/`, `logs/`, and the model cache.
- Keep `ffmpeg`, `yt-dlp`, and Python dependencies up to date.

---

Thanks again — responsible disclosure makes the whole ecosystem safer. 🙏
