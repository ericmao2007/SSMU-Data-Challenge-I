"""Build, execute, and export the explained policy notebook using the current Python."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def export_html(notebook):
    exporter = HTMLExporter(template_name='lab')
    html, _ = exporter.from_notebook_node(notebook)
    # Keep code reviewable but fold it by default in the browser report.
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    for block in soup.select('.jp-CodeCell .jp-Cell-inputWrapper'):
        details = soup.new_tag('details')
        summary = soup.new_tag('summary')
        summary.string = 'Show code'
        details.append(summary)
        block.wrap(details)
    output = ROOT / 'outputs/policy_analysis/nyc_policy_analysis.html'
    output.write_text(str(soup), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-execute', action='store_true')
    parser.add_argument('--export-only', action='store_true', help='Re-export an already executed notebook.')
    args = parser.parse_args()
    if args.export_only:
        existing = nbformat.read(ROOT / 'notebooks/nyc_policy_analysis.ipynb', as_version=4)
        print(export_html(existing))
        return
    source = (ROOT / 'notebooks/nyc_policy_analysis.py').read_text()
    sections = re.split(r'^# %%([^\n]*)\n', source, flags=re.MULTILINE)
    cells = []
    for heading, body in zip(sections[1::2], sections[2::2]):
        if '[markdown]' in heading:
            markdown = '\n'.join(line[2:] if line.startswith('# ') else
                                 '' if line == '#' else line for line in body.rstrip().splitlines())
            cells.append(nbformat.v4.new_markdown_cell(markdown))
        else:
            cells.append(nbformat.v4.new_code_cell(body.strip()))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'policy-analysis'},
        'language_info': {'name': 'python', 'version': sys.version.split()[0]},
    })
    path = ROOT / 'notebooks/nyc_policy_analysis.ipynb'
    nbformat.write(notebook, path)
    if args.no_execute:
        return
    with tempfile.TemporaryDirectory(prefix='nyc-policy-kernel-') as directory:
        kernel_path = Path(directory) / 'kernels/policy-analysis'
        kernel_path.mkdir(parents=True)
        (kernel_path / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Python 3 (policy analysis)', 'language': 'python',
        }))
        os.environ['JUPYTER_PATH'] = directory
        os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'nyc-policy-matplotlib'))
        os.environ.setdefault('IPYTHONDIR', str(Path(directory) / 'ipython'))
        client = NotebookClient(notebook, timeout=600, kernel_name='policy-analysis',
                                resources={'metadata': {'path': str(ROOT)}})
        client.on_cell_start = lambda cell, cell_index, **kw: print(
            f'Cell {cell_index + 1}/{len(cells)}: {cell.source.splitlines()[0][:95]}', flush=True)
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
    errors = [o for c in notebook.cells if c.cell_type == 'code'
              for o in c.get('outputs', []) if o.output_type == 'error']
    if errors:
        raise RuntimeError('Notebook contains execution errors')
    # Keep the previously shared normalized notebook synchronized with the source.
    nbformat.write(notebook, ROOT / 'notebooks/nyc_policy_analysis_normalized.ipynb')
    output = export_html(notebook)
    print(f'Executed notebook: {path}\nBrowser report: {output}', flush=True)


if __name__ == '__main__':
    main()
