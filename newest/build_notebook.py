from pathlib import Path
import base64
import contextlib
import io
import time
import nbformat

folder = Path(__file__).resolve().parent
source = (folder / 'nyc_crash_graphs.py').read_text()
cells = [nbformat.v4.new_code_cell(part.strip()) for part in source.split('# %%') if part.strip()]
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.14.0'}})
namespace = {'__name__': '__main__'}
started = time.monotonic()
for index, cell in enumerate(cells, 1):
    outputs = []
    def display_image(obj):
        png = obj._repr_png_()
        if isinstance(png, tuple):
            png = png[0]
        encoded = png if isinstance(png, str) else base64.b64encode(png).decode()
        outputs.append(nbformat.v4.new_output('display_data', data={
            'image/png': encoded, 'text/plain': '<PNG graph>'}, metadata={}))
    namespace['display'] = display_image
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(cell.source, str(folder / 'nyc_crash_graphs.py'), 'exec'), namespace)
    if stdout.getvalue():
        outputs.insert(0, nbformat.v4.new_output('stream', name='stdout', text=stdout.getvalue()))
    cell.outputs = outputs
    cell.execution_count = index
    print(f'Executed cell {index}/{len(cells)}', flush=True)
assert all(cell.cell_type == 'code' for cell in cells)
assert sum(o.output_type == 'display_data' for c in cells for o in c.outputs) == 8
nbformat.validate(notebook)
nbformat.write(notebook, folder / 'nyc_crash_graphs.ipynb')
print(f'Saved executed notebook and 8 graphs ({time.monotonic() - started:.1f}s)', flush=True)
