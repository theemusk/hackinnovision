"""Mission report builder: one self-contained HTML file (no extra libraries needed).
Open it in a browser; Ctrl+P -> "Save as PDF" gives a PDF copy.
"""
import base64
import html
import io
from datetime import datetime

import matplotlib.pyplot as plt

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:30px auto;padding:0 16px;color:#1c1c1c;line-height:1.5}
h1{margin-bottom:0} .sub{color:#666;margin-top:4px}
h2{border-bottom:2px solid #c0392b;padding-bottom:4px;margin-top:32px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border:1px solid #ddd;padding:6px 10px;text-align:left} th{background:#f4f4f4}
img{max-width:100%;border:1px solid #ddd;margin:6px 0} figure{margin:12px 0} figcaption{color:#555;font-size:13px}
.note{background:#fff7e6;border-left:4px solid #e67e22;padding:8px 12px;margin:10px 0}
@media print{body{margin:0}}
"""

TARGET_COLUMNS = ["rank", "target_id", "x", "y", "mission_score", "ice_probability",
                  "scientific_value", "terrain_risk", "uncertainty", "confidence"]


def _fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=105, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_report_html(key_numbers, figures, targets=None, recommendation=None, assumptions=None,
                      title="MARS-09 Mission Report", subtitle="Jezero Crater, Track A: Rover Exploration"):
    """key_numbers: list of (label, value); figures: list of (caption, matplotlib figure);
    targets: DataFrame or None; recommendation: text or None; assumptions: list of strings."""
    e = html.escape
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>{e(title)}</title>"
           f"<style>{CSS}</style></head><body>",
           f"<h1>{e(title)}</h1><p class='sub'>{e(subtitle)} &middot; generated "
           f"{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"]

    out.append("<h2>Key numbers</h2><table><tr><th>Item</th><th>Value</th></tr>")
    out += [f"<tr><td>{e(str(k))}</td><td>{e(str(v))}</td></tr>" for k, v in key_numbers]
    out.append("</table>")

    if recommendation:
        out.append(f"<h2>Mission recommendation</h2><div class='note'>{e(recommendation).replace(chr(10), '<br>')}</div>")

    if figures:
        out.append("<h2>Maps</h2>")
        for caption, fig in figures:
            out.append(f"<figure><img alt='{e(caption)}' src='data:image/png;base64,{_fig_to_img(fig)}'>"
                       f"<figcaption>{e(caption)}</figcaption></figure>")

    if targets is not None and len(targets):
        cols = [c for c in TARGET_COLUMNS if c in targets.columns] or list(targets.columns)
        top = targets[cols].head(10)
        out.append("<h2>Top science targets</h2>")
        out.append(top.to_html(index=False, border=0, float_format=lambda v: f"{v:.3f}"))

    if assumptions:
        out.append("<h2>Assumptions and limitations</h2><ul>")
        out += [f"<li>{e(a)}</li>" for a in assumptions]
        out.append("</ul>")

    out.append("<p class='sub'>Data: NASA/JPL MRO CTX mosaic and CTX DEM of Jezero Crater "
               "(USGS Astrogeology, via NASA Mars Trek).</p></body></html>")
    return "\n".join(out)