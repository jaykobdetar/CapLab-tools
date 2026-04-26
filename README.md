# CapLab-tools

Various tools for Capitalism Lab that I personally find useful, but that don't make sense to give their own repo. Tested against v11.1.2.

## Tools

- **Map Viewer** — Extracts maps from a save file as JSON, which can be loaded into `map_viewer.html` to view and verify the data.

- **focus_patch.py** linux only tool, patches a byte of memory in the running game so that it doesn't pause upon losing focus, allowing the game to run in the background, run with 'python focus_patch.py apply'

## Libraries

- **caplab_sim**, **caplab_save** — Shared library packages used by the tools above.
