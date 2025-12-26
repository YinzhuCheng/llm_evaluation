# LLM QA Playback Viewer (Static)

This is a **zero-install**, **static** (no Python) playback UI for reviewing `results_*.jsonl`.

## Quick start

1. Open `replay_viewer/index.html` in a modern browser (Chrome/Edge recommended).
2. Click **Results JSONL** and select a `results_*.jsonl` file.
3. (Optional) Put images under `replay_viewer/images/` named as `<id>.png` / `<id>.jpg` / `<id>.jpeg` / `<id>.webp` / `<id>.gif`.
4. Use **Prev/Next** (hold to fast-skip) and **Jump to ID**.
5. Write **Expert Commentary**, optionally edit question/response for paper, then:
   - **Export annotations.json** to save your edits
   - **Export PDF (Print)** to generate a PDF matching the page style

## MathJax + LLM fix (with undo)

- Math is rendered via **MathJax 3 (CDN)**.
- If rendering issues exist, click **Fix MathJax Errors (LLM)**.
- If direct calls fail due to **CORS**, switch to **Manual** mode:
  - Copy the generated `curl` command
  - Run it with your API access
  - Paste the fixed text back
- Click **Undo last fix** to revert the most recent fix.

## Notes

- This viewer intentionally shows **readable fields only** and does not display full request/response archives.
- Local edits are kept in-memory in the browser until you export `annotations.json`.

