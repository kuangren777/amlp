# Third-party projects

The baseline ports in `code/baselines/` adapt or call the projects below. Only their license texts are included here.
Upstream source is not redistributed.

| Project | Used by | Upstream URL | Commit | License | File here |
|---|---|---|---|---|---|
| Progent (`sunblaze-ucb/progent`, directories `secagent/` and its bundled AgentDojo fork) | `progent_port.py`, `progent_ab_*.py`, `progent_fidelity.py` | https://github.com/sunblaze-ucb/progent (read from the `origin` remote in `.git/config` of the checkout we used) | `5be7b63fa96f70bc19b72fbcee81f1c0bcc1a565` (shallow clone of `main`, from `.git/HEAD`/`shallow`) | MIT, "Copyright (c) 2026 Progent" | `progent_LICENSE` |
| AgentDojo (benchmark, suites v1.2, package version 0.1.35 used by the harness) | `harness.py` and everything that builds task environments | https://github.com/ethz-spylab/agentdojo (from the package metadata) | not recoverable (local checkout, installed as package 0.1.35) | MIT, "Copyright (c) 2024 Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian Tramer" | `agentdojo_LICENSE` (identical for the 0.1.35 package and for the 0.1.29 fork bundled in the Progent checkout) |
| CaMeL (`camel-prompt-injection`, arXiv 2503.18813, installed as package `camel` 1.0.0) | `camel_run.py`, `camel_fidelity.py` | https://github.com/google-research/camel-prompt-injection (from `baselines/camel_PORT_NOTES.md`; the installed copy carries no `.git` and no URL metadata) | not recoverable (installed from a copy of the clone, `direct_url.json` has only a local path) | Apache License 2.0 (no NOTICE file in the package) | `camel-prompt-injection_LICENSE` |

Re-implementations without upstream code
- Agent-Sentry (arXiv 2603.22868): `agentsentry_port.py` is a re-implementation from the paper text, no public code was found.
- ToolFence (arXiv 2609.37196): `toolfence_port.py` is a re-implementation from the paper text.
- MELON and TripWire arms referenced in `harness.py` come from a separate code base that is not part of this release.
  Set `TRIPWIRE_DIR` to a checkout that provides `tripwire.py` and `melon_port.py` to enable them.

To rerun the Progent and CaMeL baselines, clone the upstream repositories yourself, check out the commit above for
Progent, and point `CAMEL_REPO` (for CaMeL) at the checkout. The ports expect, for Progent, a clone of the repository
at `code/baselines/progent_up_env/clone` (the directory name used in our development tree), and a Python 3.12
virtual environment for CaMeL (`baselines/camel_PORT_NOTES.md` is not shipped, the installed versions are
`pydantic-ai==0.2.12`, `openai==1.82.1`, `pydantic==2.11.5`).
