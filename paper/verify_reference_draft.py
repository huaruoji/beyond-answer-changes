#!/usr/bin/env python3
"""Read-only evidence and PDF checks for reference-draft.tex.

Run from any directory with Python 3. PDF checks additionally require pypdf.
No inference, waveform generation, or frozen-artifact updates are performed.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

# The paper kit inspects the uploaded file against the web submission form, so
# this string must stay identical to the title typed into the online form. It
# carries a colon even though the prose avoids colons.
REGISTERED_TITLE = (
    "BEYOND ANSWER CHANGES: MATCHED SAME-CLASS CONTROLS FOR LOCAL AUDIO EVIDENCE"
)


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-12), (a, b)


def main():
    table = json.loads((ROOT / "outputs/evidence/evidence_table.json").read_text())
    for path, expected in table["sources_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    print(f"Frozen source hashes: {len(table['sources_sha256'])} verified")
    for co, cd, cs in itertools.product((0, 1), repeat=3):
        # Original/same truth is A; different-class truth is B.
        ao, ad, ass = co, 1 - cd, cs
        assert int(ao != ass) == co + cs - 2 * co * cs
        assert int(ao != ad) == 1 - co - cd + 2 * co * cd
    print("Binary identities: all eight states verified")

    for key in ("r2_qwen7b", "r2_qwen3b", "fun_audio_chat_8b"):
        m = table["models"][key]
        f, a = m["follow"], m["attribution"]
        views = defaultdict(dict)
        for line in (ROOT / f["predictions"]).read_text().splitlines():
            r = json.loads(line)
            if r.get("split") != "confirm":
                continue
            assert r["condition"] not in views[r["pair_id"]], "duplicate prediction"
            assert r["parsed_choice"] in ("A", "B"), "nonbinary prediction"
            assert r["correct"] == (r["parsed_choice"] == r["ground_truth"])
            views[r["pair_id"]][r["condition"]] = r
        assert len(views) == 1200
        conditions = tuple(a["accuracy_by_condition"])
        assert all(set(v) == set(conditions) for v in views.values())
        for c in conditions:
            close(sum(v[c]["correct"] for v in views.values()) / len(views),
                  a["accuracy_by_condition"][c])
        by_pair = defaultdict(list)
        eligible = []
        correct = []
        changes = defaultdict(list)
        for v in views.values():
            o, d, s = (v[c] for c in ("original", "target_replaced_diff", "target_replaced_same"))
            assert o["option_events"] == d["option_events"] == s["option_events"]
            assert o["ground_truth"] == s["ground_truth"] != d["ground_truth"]
            by_pair[o["source_pair_id"]].append(v)
            correct.append((int(o["correct"]), int(d["correct"]), int(s["correct"])))
            for c in conditions:
                changes[c].append(int(v[c]["parsed_choice"] != o["parsed_choice"]))
            if o["correct"] and s["correct"]:
                eligible.append(v)
        assert len(by_pair) == 600
        assert all({v["original"]["surface_order"] for v in vs} == {0, 1} for vs in by_pair.values())
        for c, field in (("target_replaced_diff", "A_diff"), ("target_replaced_same", "A_same"),
                         ("nuisance_removed", "A_nuisance_removed"), ("sham", "A_sham"),
                         ("background_removed", "A_background_removed"), ("silence", "A_silence")):
            close(sum(changes[c]) / 1200, a[field])
        e2 = f["subsets"]["E2_orig_and_same"]
        assert len(eligible) == e2["n"]
        close(sum(v["target_replaced_diff"]["correct"] for v in eligible) / len(eligible),
              e2["follow_diff"]["rate"])
        pair_values = []
        for vs in by_pair.values():
            selected = [v for v in vs if v["original"]["correct"] and v["target_replaced_same"]["correct"]]
            if selected:
                pair_values.append(sum(v["target_replaced_diff"]["correct"] for v in selected) / len(selected))
        pm = f["pair_mean"]["E2_orig_and_same"]
        assert len(pair_values) == pm["n_source_pairs"]
        close(sum(pair_values) / len(pair_values), pm["follow_diff"])
        ao, ad, ass = [sum(r[i] for r in correct) / 1200 for i in range(3)]
        reference = (2 * ao - 1) * (ad + ass - 1)
        cov_d = sum(r[0] * r[1] for r in correct) / 1200 - ao * ad
        cov_s = sum(r[0] * r[2] for r in correct) / 1200 - ao * ass
        close(reference + 2 * cov_d + 2 * cov_s, a["selectivity"])
        print(f"{m['display']}: 8400 records checked; E2={e2['follow_diff']['rate']:.6f} "
              f"({len(eligible)} trials), pair mean={pm['follow_diff']:.6f} "
              f"({len(pair_values)} items); G_ind={reference:.6f}; covariance identity verified")

    tex = (PAPER / "reference-draft.tex").read_text()
    abstract = tex.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
    word_count = len(abstract.split())
    assert 100 <= word_count <= 150, word_count
    keywords = tex.split(r"\begin{keywords}")[1].split(r"\end{keywords}")[0]
    assert len(keywords.split(",")) <= 5
    chapters = sorted((PAPER / "chapters").glob("*.tex"))
    assert len(chapters) == 6, [p.name for p in chapters]
    body = tex + "\n".join(p.read_text() for p in chapters)
    cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", body) for key in group.split(",")}
    bib = (PAPER / "reference-refs.bib").read_text()
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
    assert cited <= bibkeys, cited - bibkeys
    print(f"Abstract: {word_count} words; {len(chapters)} chapters; {len(cited)} citation keys resolved")

    # The manuscript's headline numbers must appear literally, so a stale draft
    # cannot ship after the frozen table moves.
    for key in ("r2_qwen7b", "r2_qwen3b", "fun_audio_chat_8b"):
        m = table["models"][key]
        follow, attribution = m["follow"], m["attribution"]
        gap = attribution["selectivity"] * 100
        low, high = (v * 100 for v in attribution["selectivity_ci95"])
        assert f"{gap:.2f}" in body, (key, "answer-change gap missing", gap)
        assert f"{low:.2f}, {high:.2f}" in body, (key, "answer-change interval missing", low, high)
        same = follow["subsets"]["all"]["preserve_same"]["rate"]
        assert f"{same:.3f}" in body, (key, "same-class accuracy missing", same)
        assert m["display"] in body, m["display"]
    # The abstract quotes the spread of the contrast, the sham ceiling, and the
    # same-class recognition cost; all three must match the table.
    gaps = sorted(table["models"][k]["attribution"]["selectivity"] * 100
                  for k in ("r2_qwen7b", "r2_qwen3b", "fun_audio_chat_8b"))
    assert f"{gaps[0]:.1f} to {gaps[-1]:.1f}" in abstract, ("abstract gap range", gaps)
    sham = max(table["models"][k]["attribution"]["A_sham"]
               for k in ("r2_qwen7b", "r2_qwen3b", "fun_audio_chat_8b"))
    assert sham < 0.010, sham
    worst = max(
        table["models"][k]["attribution"]["original_accuracy"]
        - table["models"][k]["follow"]["subsets"]["all"]["preserve_same"]["rate"]
        for k in ("r2_qwen7b", "r2_qwen3b", "fun_audio_chat_8b")
    ) * 100
    assert f"{worst:.1f}" in abstract, ("same-class recognition cost", worst)
    print(f"Manuscript numbers: contrast {gaps[0]:.1f}-{gaps[-1]:.1f} pp, "
          f"sham ceiling {sham:.4f}, same-class cost {worst:.1f} pt -- all present")

    pdf = PAPER / "reference-draft.pdf"
    if pdf.exists():
        from pypdf import PdfReader
        reader = PdfReader(pdf)
        assert not reader.is_encrypted
        assert len(reader.pages) <= 5
        assert pdf.stat().st_size <= 5_000_000
        for index, page in enumerate(reader.pages, 1):
            close(float(page.mediabox.width), 612)
            close(float(page.mediabox.height), 792)
            fonts = page["/Resources"].get("/Font", {})
            for ref in fonts.values():
                font = ref.get_object()
                assert font["/Subtype"] != "/Type3"
                descendants = font.get("/DescendantFonts", [font])
                for child in descendants:
                    child = child.get_object()
                    desc = child.get("/FontDescriptor")
                    assert desc is not None, font
                    desc = desc.get_object()
                    assert any(k in desc for k in ("/FontFile", "/FontFile2", "/FontFile3")), font
            text = page.extract_text()
            print(f"PDF page {index}: {len(text.split())} extracted words; Letter; embedded non-Type3 fonts")
        # All technical sections must end by page four. The conclusion is the last
        # numbered section, so match its number rather than a hard-coded "5.".
        pattern = re.compile(r"\b\d+\.\s*CONCLUSION\b")
        last_body = next(
            i for i, p in enumerate(reader.pages, 1) if pattern.search(p.extract_text())
        )
        assert last_body <= 4, f"conclusion landed on page {last_body}"

        pages_text = [p.extract_text() for p in reader.pages]
        ack = next((i for i, t in enumerate(pages_text, 1)
                    if re.search(r"^\s*Acknowledgments\s*$", t, re.M)), None)
        assert ack is not None, "no Acknowledgments section in the PDF"
        # The paper kit allows funding acknowledgements and the compliance
        # statement on page five, so the acknowledgement block follows the
        # technical content and page four stays technical.
        assert ack >= 5, f"Acknowledgments landed on page {ack}, before the end of the technical content"
        assert len(pages_text) >= 5
        assert re.search(r"no external funding", pages_text[4], re.I)

        # The title on the uploaded file must match the title typed into the web
        # submission form. A mismatch fails document inspection, so guard it.
        # Whitespace is stripped before comparing because PDF text extraction
        # inserts stray spaces inside words and at line breaks.
        first_page = re.sub(r"\s+", "", pages_text[0]).upper()
        assert REGISTERED_TITLE.replace(" ", "") in first_page, (
            "PDF title does not match the online submission form",
            REGISTERED_TITLE,
        )

        # Page five may hold references and a Compliance statement.
        # No numbered technical section may appear there.
        if len(pages_text) >= 5:
            headings = re.findall(r"^\s*\d+\.\s*[A-Z][A-Z ]+$", pages_text[4], re.M)
            stray = [h.strip() for h in headings if "REFERENCES" not in h]
            assert not stray, f"page 5 carries technical sections: {stray}"

        print(f"PDF: {len(reader.pages)} pages, {pdf.stat().st_size} bytes, "
              f"conclusion on page {last_body}, acknowledgments on page {ack}. "
              f"Manual content/layout review still required.")


if __name__ == "__main__":
    main()
