# Contributing

Issues and pull requests are welcome.

- The backend is **Python 3.9+ standard library only** — no pip dependencies.
  Please keep it that way unless a change justifies otherwise.
- The backend lives in `b70launcher/` (`paths`, `locks`, `catalog`, `scan`,
  `hardware`, `downloads`, `builders/`, `runtime`, `telemetry`, `harness`,
  `http_api`, `exl3_stack`); `launcher.py` is the desktop entry point and
  `cli.py` the headless one. Adding behavior means adding it to the package,
  not to an entry point, and importing a module must stay side-effect free —
  state setup happens in `paths.init_state()` from the entry point.
- Run the suite before opening a PR: `python3 -m unittest discover -s tests -v`
- Keep the two public references in step with the code: `docs/api.md` mirrors
  the route tables in `b70launcher/http_api.py`, and `docs/recipe-format.md`
  mirrors the recipe fields the package actually reads.
- Do not commit machine-specific paths, tokens, or state — the tarball must stay
  portable. See `.gitignore` and `packaging/release.py` for what ships.
- Recipe changes that alter launch behavior must bump the recipe's
  `recipe_ver` (the `ver` the published manifest carries) so installed
  launchers show the update notice. Display-only edits do not.
- Security issues: see [SECURITY.md](SECURITY.md) — report privately.
