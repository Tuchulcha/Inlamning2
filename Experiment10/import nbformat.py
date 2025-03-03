import nbformat
import pyperclip
import os
print(os.getcwd())


def copy_notebook_cells(file_path):
    # Load the Jupyter Notebook
    notebook = nbformat.read(file_path, as_version=4)

    # Extract the content of each cell
    cell_contents = []
    for cell in notebook.cells:
        cell_contents.append(cell['source'])

    # Join all cell contents into a single string
    all_content = '\n\n'.join(cell_contents)

    # Copy to clipboard (you can also print or save to a file)
    pyperclip.copy(all_content)
    print("Content copied to clipboard.")

# Replace 'your_notebook.ipynb' with the path to your notebook
copy_notebook_cells('C:\BIA\Kurs8\Inlamning2\Experiment10\Exp10_small_tnn.ipynb')
