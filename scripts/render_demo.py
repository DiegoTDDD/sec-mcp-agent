"""Render an animated terminal demo (GIF and MP4) from a real eval answer.

The frames are drawn from eval/results.json: the question, the tool calls the agent made with
their arguments, the answer and its verified citations. Nothing is typed by hand, so the
demo shows exactly what the evaluation recorded.

    python scripts/render_demo.py f02 t01      # writes docs/demo.gif and docs/demo.mp4
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
COLS, ROWS, SIZE, PAD = 96, 24, 15, 18
BG, FG = (13, 17, 23), (201, 209, 217)
GREEN, CYAN, YELLOW, GRAY, RED = (
    (63, 185, 80),
    (86, 182, 194),
    (210, 168, 75),
    (125, 133, 144),
    (248, 81, 73),
)

Line = tuple[str, tuple[int, int, int]]


def wrap(text: str, color: tuple[int, int, int], indent: str = "") -> list[Line]:
    out: list[Line] = []
    for para in text.splitlines() or [""]:
        for chunk in textwrap.wrap(para, COLS - len(indent)) or [""]:
            out.append((indent + chunk, color))
    return out


def script_for(row: dict) -> tuple[str, list[list[Line]]]:
    """The typed command, then blocks of lines revealed one after another."""
    a = row["answer"]
    command = f'$ python -m sec_mcp_agent "{a["question"]}"'
    blocks: list[list[Line]] = []
    for t in a["tool_calls"]:
        args = json.dumps(t["arguments"], ensure_ascii=False)
        mark = "" if t["ok"] else "  (error)"
        blocks.append(wrap(f"→ {t['server']}.{t['tool']} {args}{mark}", CYAN, "  "))
    blocks.append([("", FG), *wrap(a["answer"], FG)])
    for i, c in enumerate(a["citations"], 1):
        head = f"[{i}] ✓ {c['company']} ({c['ticker']}) 10-K {c['filing_date']}, {c['section']}"
        blocks.append([("", FG), (head, YELLOW), *wrap(f'"{c["quote"]}"', GRAY, "    ")])
    for r in a["rejected_citations"]:
        blocks.append(wrap(f"[rejected] {r['chunk_id']}: {r['reason']}", RED))
    for s in a["fraud_scores"]:
        reasons = ", ".join(f"{x['feature']} {x['log_odds']:+.2f}" for x in s["top_reasons"])
        blocks.append(
            [("", FG), *wrap(f"fraud model: p={s['probability']:.3f}, flagged={s['flagged']} "
                             f"(threshold {s['threshold']:.3f}); reasons: {reasons}", YELLOW)]
        )  # fmt: skip
    n = len(a["tool_calls"])
    footer = (f"grounded={a['grounded']} · {n} tool call{'s' * (n != 1)} · {a['model']} · "
              f"{a['seconds']:.0f}s on a CPU runner")  # fmt: skip
    blocks.append([("", FG), (footer, GREEN if a["grounded"] else RED)])
    return command, blocks


class Screen:
    def __init__(self) -> None:
        self.font = ImageFont.truetype(FONT, SIZE)
        box = self.font.getbbox("M")
        self.cw, self.ch = box[2] - box[0], int(SIZE * 1.45)
        self.w, self.h = COLS * self.cw + 2 * PAD, ROWS * self.ch + 2 * PAD
        self.frames: list[tuple[Image.Image, int]] = []

    def draw(self, lines: list[Line], ms: int) -> None:
        img = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(img)
        for i, (text, color) in enumerate(lines[-ROWS:]):
            d.text((PAD, PAD + i * self.ch), text, font=self.font, fill=color)
        self.frames.append((img, ms))


def render(rows: list[dict], gif: Path, mp4: Path) -> None:
    screen = Screen()
    for n, row in enumerate(rows):
        command, blocks = script_for(row)
        typed = wrap(command, GREEN)
        screen.draw([], 400)
        flat = "\n".join(t for t, _ in typed)
        for i in range(0, len(flat) + 1, 4):  # type the command
            screen.draw(wrap(flat[:i] + "▌", GREEN), 35)
        lines = typed[:]
        screen.draw(lines, 500)
        for block in blocks:
            lines += block
            screen.draw(lines, 700)
        screen.draw(lines, 4500 if n == len(rows) - 1 else 3000)

    gif.parent.mkdir(parents=True, exist_ok=True)
    images = [f for f, _ in screen.frames]
    images[0].save(gif, save_all=True, append_images=images[1:], loop=0, optimize=True,
                   duration=[ms for _, ms in screen.frames])  # fmt: skip
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as tmp:
            concat = Path(tmp) / "frames.txt"
            entries = []
            for i, (img, ms) in enumerate(screen.frames):
                path = Path(tmp) / f"{i:05d}.png"
                img.save(path)
                entries.append(f"file '{path}'\nduration {ms / 1000:.3f}")
            entries.append(f"file '{path}'")
            concat.write_text("\n".join(entries), encoding="utf-8")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                 str(concat), "-vf", "fps=25,format=yuv420p", "-c:v", "libx264", "-preset",
                 "veryfast", "-crf", "23", str(mp4)],
                check=True,
            )  # fmt: skip


def main() -> None:
    ids = sys.argv[1:] or ["f02", "t01"]
    results = json.loads((ROOT / "eval" / "results.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in results["rows"]}
    render([by_id[i] for i in ids], ROOT / "docs" / "demo.gif", ROOT / "docs" / "demo.mp4")
    print("wrote docs/demo.gif and docs/demo.mp4")


if __name__ == "__main__":
    main()
