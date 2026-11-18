# molass_legacy changelog

Newest first. This file ships inside the installed package, next to `CONTEXT.md`. It is the engine under `molass`
(`pip install molass`), which is what you call; most users never import `molass_legacy` directly. Issue numbers are of
`biosaxs-dev/molass-legacy` unless written out. Versions before 1.7.2: see the git tags.

Headings: **Changed (behavior)**, **Added**, **Fixed**, **Requires**.

## 1.7.3 (2026-11-18)

### Changed (behavior)
- Nothing changes for existing code or results. The manifest progress below needs `molass` 1.3.0: the call is made only when the
  installed `molass` has the option, so an older `molass` is unaffected.

### Added
- `MOLASS_SETTINGS_NAME` (environment variable) overrides the script name that `BasicUtils.exe_name()` uses for the settings
  folder, so that parallel runs do not share one (biosaxs-dev/molass-library#393).
- The optimizer subprocess (`optimizer_recipe.py`) keeps `RUN_MANIFEST.json` current while it runs, through
  `create_optimizer_from_recipe(..., watch_manifest=True)` of `molass` (biosaxs-dev/molass-library#397).

### Fixed
- `FlangeLimit`: no `UnboundLocalError` when no flange candidate is found (#104).

### Requires
`molass>=1.1.2`; `molass` 1.3.0 or later for the manifest progress.

## 1.7.2 (2026-10-30)

### Changed (behavior)
- The EGH `tau` and EDM `b` bounds follow `fronting_flags` (one bool per component, set by `molass`): only a flagged component may
  front (biosaxs-dev/molass-library#350). Recipes written before the policy was recorded keep the former bounds.

### Fixed
- `Edm.edm_func` skips points with `tau/RPe <= 0` in the nonlinear branch (biosaxs-dev/molass-library#341); the `Edm` class
  documents that it fails for `b = 0` and which functions handle it.
