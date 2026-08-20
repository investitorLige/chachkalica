"""Render the admin Benchmark Console as ONE self-contained HTML page.

The console proper lives in the Django admin (``templates/admin/benchmarks/console.html``
plus one ``<image_size>.json`` per size in ``chachkalica/data/benchmarks/``), where the
size switcher is a server-side ``?size=`` round trip. A published copy of that table has
to work with no Django, no admin session and no network, so this script rewrites the same
template into a single file: every size envelope inlined, the switcher moved to the
client, and the handful of Django tags resolved.

It deliberately REUSES the template's own CSS and rendering JS rather than re-styling a
second page -- the published table should be the same table, not a lookalike that drifts
the next time a column changes.

    python inferlica/benchmark/gen_console_artifact.py \
        --template chachkalica/templates/admin/benchmarks/console.html \
        --data chachkalica/data/benchmarks \
        --out /tmp/benchmark_console.html

The output has no ``<!doctype>``/``<html>``/``<head>``/``<body>`` wrapper: it is meant to
be published with the Artifact tool, which supplies that skeleton (it keeps a leading
``<title>``). Open it directly in a browser for a quick check anyway -- browsers parse a
bare fragment fine.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _extract(text: str, start: str, end: str) -> str:
    a = text.index(start) + len(start)
    b = text.index(end, a)
    return text[a:b]


def build(template_path: Path, data_dir: Path, title: str) -> str:
    template = template_path.read_text()

    css = _extract(template, "<style>", "</style>")
    body = _extract(template, "<body>", "</body>")
    js = _extract(body, "<script>", "</script>")

    envelopes = []
    for path in sorted(data_dir.glob("*.json"), key=lambda p: int(p.stem)):
        envelope = json.loads(path.read_text())
        envelopes.append(envelope)
    if not envelopes:
        raise SystemExit(f"no <size>.json envelopes in {data_dir}")

    # Django tags, all of them, resolved against the envelopes instead of a request.
    # The header/footer values become spans the switcher rewrites; the size tabs become
    # buttons; the admin backlink has no meaning off the admin and goes away.
    header = _extract(body, '<div class="meta-row">', "</div>")
    new_header = (
        '\n      <span>device <b id="meta-device"></b></span>\n'
        '      <span>image size <b id="meta-size"></b></span>\n'
        '      <span id="meta-rfdetr-wrap">rfdetr <b id="meta-rfdetr"></b></span>\n'
        "      <span>util <b>CUPTI</b></span>\n    "
    )
    body = body.replace(header, new_header)

    tabs = _extract(body, '<span class="lbl">size</span>', "</div>")
    body = body.replace(
        tabs, '\n        <span id="sizetabs"></span>\n      '
    )
    body = re.sub(r'\s*<a class="backlink"[^>]*>.*?</a>', "", body, flags=re.S)

    body = body.replace(
        '{{ bench_data|json_script:"bench-data" }}',
        '<script id="bench-sizes" type="application/json">'
        + json.dumps(envelopes, separators=(",", ":"))
        + "</script>",
    )
    body = body.replace(
        "{% if bench.generated %} &middot; {{ bench.generated }}{% endif %}",
        ' &middot; <span id="meta-generated"></span>',
    )
    body = body.replace("{% verbatim %}", "").replace("{% endverbatim %}", "")

    # The template's JS reads one envelope's `data` from a json_script tag and renders
    # once at load. Here it gets a selectable list, so `DATA` becomes a mutable current
    # size and the render bootstrap moves into a function the tabs call.
    js = js.replace(
        "const DATA = JSON.parse(document.getElementById('bench-data').textContent);",
        "const SIZES = JSON.parse(document.getElementById('bench-sizes').textContent);\n"
        "let DATA = SIZES[0].data;",
    )
    bootstrap_start = js.index("const container = document.getElementById('tables');")
    arch_order = _extract(js[bootstrap_start:], "[", "]")
    js = js[:bootstrap_start] + f"""const ARCH_ORDER = [{arch_order}];
const container = document.getElementById('tables');

function selectSize(index) {{
  const envelope = SIZES[index];
  DATA = envelope.data;
  document.getElementById('meta-device').textContent = envelope.device || 'cuda';
  document.getElementById('meta-size').textContent = envelope.label || envelope.image_size;
  document.getElementById('meta-generated').textContent = envelope.generated || '';
  const rfdetr = document.getElementById('meta-rfdetr');
  rfdetr.textContent = envelope.rfdetr_note || '';
  document.getElementById('meta-rfdetr-wrap').style.display = envelope.rfdetr_note ? '' : 'none';
  document.querySelectorAll('#sizetabs a').forEach((a, i) => {{
    a.className = i === index ? 'active' : '';
  }});
  container.innerHTML = '';
  ARCH_ORDER.forEach(a => {{
    const el = renderArch(a);
    if (el) container.appendChild(el);
  }});
}}

document.getElementById('sizetabs').innerHTML = SIZES.map(
  (s, i) => `<a href="#" data-index="${{i}}">${{s.label || s.image_size}}</a>`
).join('');
document.querySelectorAll('#sizetabs a').forEach(a => {{
  a.addEventListener('click', event => {{
    event.preventDefault();
    selectSize(Number(a.dataset.index));
  }});
}});

// 640 is the reference size the other two are read against, same default as the view.
const initial = SIZES.findIndex(s => s.image_size === 640);
selectSize(initial >= 0 ? initial : 0);
</script>"""

    body = body[: body.index("<script>")] + "<script>" + js
    return f"<title>{title}</title>\n<style>{css}</style>\n{body}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template", type=Path,
        default=Path("chachkalica/templates/admin/benchmarks/console.html"),
    )
    parser.add_argument("--data", type=Path, default=Path("chachkalica/data/benchmarks"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Benchmark Console")
    args = parser.parse_args()

    html = build(args.template, args.data, args.title)
    args.out.write_text(html)
    print(f"[artifact] wrote {args.out} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
