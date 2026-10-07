# Contributing

Issues and pull requests are welcome.

- The backend is **Python 3.9+ standard library only** — no pip dependencies.
  Please keep it that way unless a change justifies otherwise.
- Run the suite before opening a PR: `python3 -m unittest discover -s tests -v`
- Do not commit machine-specific paths, tokens, or state — the tarball must stay
  portable. See `.gitignore` and `packaging/release.py` for what ships.
- Recipe changes that alter launch behavior must bump the recipe `ver` field so
  installed launchers show the update notice. Display-only edits do not.
- Security issues: see [SECURITY.md](SECURITY.md) — report privately.
