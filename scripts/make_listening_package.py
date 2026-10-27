#!/usr/bin/env python3
"""Build a listening package for the human sanity check.

Synthetic ground truth is exact by construction, so listening is not checking
labels. It is checking that the instrument is perceptually sane: that the
background is audible, the query event is clearly audible at its calibrated SNR,
no edit left a click or a seam, and the two option strings are both plausible
labels for the sounds they name. A systematic defect found here sends the design
back to the mixer rather than into a caveat.

The output is a self-contained HTML page with an audio player per condition and
the checks laid out next to each item, plus the audio it references. Open
``index.html`` in a browser; no server needed.

    python scripts/make_listening_package.py \
        --manifest data/processed/synth/pilot.jsonl \
        --audio-root data/processed/synth/pilot_audio \
        --output data/processed/listening --items 20 --seed 20260916
"""

from __future__ import annotations

import argparse
import html
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

#: Conditions worth listening to. The rest are the same edits with different
#: fill, and the point is to check boundaries and audibility, not to audit every
#: cell by ear.
LISTEN_CONDITIONS = (
    "original",
    "target_replaced_diff",
    "nuisance_removed",
    "background_removed",
    "sham",
)

#: Chinese label shown next to the raw condition name, so the page can be read
#: without translating the identifiers the manifest and analysis use.
CONDITION_LABELS = {
    "original": "原始（未编辑）",
    "target_replaced_diff": "替换成异类声音",
    "nuisance_removed": "移除干扰声音",
    "background_removed": "移除背景",
    "sham": "空操作（应与原始相同）",
}

PROMPTS = {
    "original": "背景能听见吗？被查询的声音清楚吗？有削顶吗？",
    "target_replaced_diff": "替换上去的声音清楚吗？两个边界有咔哒声或接缝吗？",
    "nuisance_removed": "能听出被挖掉一块吗？两个边界有咔哒声或接缝吗？",
    "background_removed": "听起来像一段没编辑过的音频吗？有咔哒声或接缝吗？",
    "sham": "听起来和「原始」一样吗？（应该一样，最多差在舍入。）",
}

CHECKS = (
    "背景能听见，且没有削顶",
    "被查询的声音清楚可辨",
    "任何编辑边界都没有咔哒声或接缝",
    # Added after a listener (2026-09-21) caught something none of the original
    # five covered: the substituted sound had no noise floor, so it sat in a
    # different acoustic space from the rest of the clip. That was a real
    # instrument defect, not a perception error, and it was invisible to every
    # check above -- the boundaries were continuous and the event was audible.
    "替换/删除的声音和周围环境音融合（不会突然没了底噪）",
    "两个选项的文字都说得通",
    "另一个事件不会和两个选项中的任何一个混淆",
)

#: Injected verbatim into the page. It exists because the first listening session
#: (2026-09-18) left no per-item record -- only a prose summary in the handoff --
#: so a second listener could not be compared against it. The export carries the
#: listener name, the seed the item set was drawn with, and one row per item so
#: two sessions can be diffed.
_EXPORT_SCRIPT = """
<script>
function collect() {
  const listener = (document.getElementById('listener').value || '').trim();
  const items = [...document.querySelectorAll('section')].map(sec => {
    const boxes = [...sec.querySelectorAll('.checks input[type=checkbox]')];
    const note = sec.querySelector('.notes input');
    return {
      item: sec.dataset.item,
      number: Number(sec.dataset.number),
      checks: boxes.map(b => ({label: b.dataset.check, checked: b.checked})),
      note: note ? note.value : ''
    };
  });
  return {listener, seed: SEED, exported: new Date().toISOString(), items};
}
function download(name, text) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {type: 'application/json'}));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}
function exportResults() {
  const data = collect();
  const who = (data.listener || 'anonymous').replace(/[^\\w.-]+/g, '_');
  download('listening-' + who + '.json', JSON.stringify(data, null, 2));
  document.getElementById('status').textContent = '已下载 listening-' + who + '.json';
}
function copyResults() {
  const text = JSON.stringify(collect(), null, 2);
  navigator.clipboard.writeText(text).then(
    () => document.getElementById('status').textContent = '已复制，粘贴到文件即可',
    () => document.getElementById('status').textContent = '复制失败，请用「下载 JSON」'
  );
}
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]
    by_item: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_item[row["source_pair_id"]][row["condition"]] = row

    eligible = [
        item
        for item, conditions in by_item.items()
        if all(condition in conditions for condition in LISTEN_CONDITIONS)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    chosen = sorted(eligible[: args.items])
    if len(chosen) < args.items:
        print(f"note: only {len(chosen)} items have every listenable condition")

    args.output.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output / "audio"
    audio_dir.mkdir(exist_ok=True)

    sections: list[str] = []
    index: list[dict] = []
    for number, item in enumerate(chosen, start=1):
        conditions = by_item[item]
        original = conditions["original"]
        options = original["option_events"]
        blocks: list[str] = []
        entry = {
            "item": item,
            "number": number,
            "options": options,
            "query_span_s": original.get("query_span_s"),
            "query_start_s": original.get("query_start_s"),
            "snr_db": original.get("snr_db"),
            "background_class": original.get("background_class"),
            "files": {},
        }
        for condition in LISTEN_CONDITIONS:
            row = conditions[condition]
            source = args.audio_root / Path(row["audio"]).name
            if not source.exists():
                continue
            target = audio_dir / source.name
            if not target.exists():
                shutil.copy2(source, target)
            entry["files"][condition] = target.name
            blocks.append(
                f"""
        <div class="cond">
          <div class="condhead"><strong>{html.escape(CONDITION_LABELS[condition])}</strong>
            <span class="condname">{html.escape(condition)}</span>
            <span class="ask">{html.escape(PROMPTS[condition])}</span></div>
          <audio controls preload="none" src="audio/{html.escape(target.name)}"></audio>
        </div>"""
            )
        index.append(entry)
        checks_html = ''.join(
            f'<label><input type="checkbox" data-check="{html.escape(c)}"> '
            f'{html.escape(c)}</label>'
            for c in CHECKS
        )
        sections.append(
            f"""
    <section data-item="{html.escape(item)}" data-number="{number}">
      <h2>第 {number} 题 <span class="meta">{html.escape(item)}</span></h2>
      <p class="q">题目：这两个声音里，哪一个在音频中？<span class="meta">（选项文字是模型的题目原文，未翻译）</span><br>
         <b>A</b>: {html.escape(options['A'])} &nbsp;|&nbsp;
         <b>B</b>: {html.escape(options['B'])}</p>
      <p class="meta">被查询区间 {entry['query_start_s']}s 起、长 {entry['query_span_s']}s &middot;
         SNR {entry['snr_db']} dB &middot; 背景类型 {html.escape(str(entry['background_class']))}</p>
      {''.join(blocks)}
      <div class="checks">{checks_html}</div>
      <label class="notes">备注 <input type="text" placeholder="哪里不对？例如「第 6.2 秒有咔哒声」"></label>
    </section>"""
        )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>合成声景仪器听检</title>
<script>const SEED = {args.seed};</script>
<style>
 body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color:#111; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.05rem; margin-bottom: .2rem; }}
 .meta {{ color:#666; font-weight: 400; font-size: .85rem; }}
 .q {{ margin: .3rem 0; }}
 section {{ border:1px solid #ddd; border-radius:8px; padding:1rem; margin:1.2rem 0; }}
 .cond {{ margin:.6rem 0; padding:.5rem; background:#fafafa; border-radius:6px; }}
 .condhead {{ display:flex; gap:.6rem; align-items:baseline; flex-wrap:wrap; }}
 .ask {{ color:#555; font-size:.85rem; }}
 .condname {{ color:#999; font-size:.78rem; font-family:ui-monospace,monospace; }}
 audio {{ width:100%; margin-top:.35rem; }}
 .checks {{ display:grid; gap:.15rem; margin-top:.7rem; font-size:.9rem; }}
 .checks label {{ display:block; }}
 .notes {{ display:block; margin-top:.6rem; font-size:.9rem; }}
 .notes input {{ width:70%; }}
 .howto {{ background:#f0f6ff; border:1px solid #cfe0f7; border-radius:8px; padding:1rem; }}
 .export {{ position:sticky; top:0; z-index:5; background:#fffbe6; border:1px solid #e6d98a;
            border-radius:8px; padding:.8rem 1rem; margin:1rem 0; }}
 .export input {{ font:inherit; padding:.25rem .4rem; width:14rem; }}
 .export button {{ font:inherit; padding:.3rem .8rem; margin-left:.4rem; cursor:pointer; }}
 .export .status {{ color:#555; font-size:.85rem; margin-left:.6rem; }}
</style></head><body>
<h1>合成声景仪器听检</h1>
<div class="howto">
 <p><b>这是在查什么。</b>这些音频的真值是由程序精确构造的，所以<b>不是</b>在核对标签。
 查的是仪器在<b>人耳听来</b>是否正常：背景可闻、事件清晰、编辑处无瑕疵。
 这里若发现系统性缺陷，设计要退回混音环节重做，而不是写一条 caveat 糊过去。</p>
 <p><b>怎么做。</b>每题先听<b>原始</b>，再依次听各个编辑条件。符合就打勾。
 发现问题写进「备注」——「背景移除在第 6.2 秒有咔哒声」远比「听着不对」有用。</p>
 <p><b>最关键的一项。</b><b>空操作</b>是用它自己的样本把同一区间写回一遍，应当与<b>原始</b>
 听不出区别。如果你能听出差异，请停下并报告——那样这台仪器产出的其他任何数字都不能采信。</p>
 <p><b>关于「融合」这一项。</b>替换或删除过的片段，听上去应当<b>仍在同一个声学空间里</b>——
 底噪不该突然消失。如果某个片段像是从别处贴进来的（安静得不自然、和周围环境脱节），
 请指出，并写下大致秒数。2026-09-21 的一次听检正是靠这一项发现了仪器的真实缺陷。</p>
 <p><b>不需要每格都听。</b>重点是有没有咔哒/接缝、事件在标定 SNR 下是否真的听得见、
 替换片段是否仍与背景融合、两个选项是否都说得通。拿不准的记下来即可。</p>
</div>
<div class="export">
 <b>做完请导出结果</b>（勾选和备注都在这份文件里，不会上传到任何地方）：
 <input id="listener" type="text" placeholder="你的名字（用于报告一致性）">
 <button onclick="exportResults()">下载 JSON</button>
 <button onclick="copyResults()">复制到剪贴板</button>
 <span class="status" id="status"></span>
</div>
{''.join(sections)}
</body></html>
"""
    (args.output / "index.html").write_text(page + _EXPORT_SCRIPT)
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "items": len(chosen),
                "audio_files": len(list(audio_dir.iterdir())),
                "open_this": str(args.output / "index.html"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
