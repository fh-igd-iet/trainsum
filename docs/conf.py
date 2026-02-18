# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'trainsum'
copyright = '2025-2026 Fraunhofer Institute for Computer Graphics Research IGD'
author = 'Paul Haubenwallner'
release = '2026'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinx_copybutton'
]

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_logo = 'pics/logo.png'
html_static_path = ['_static']
html_theme_options = {
    'logo_only': True,
}

# -- Custom code to replace "no title" with the actual title in the generated HTML files

def get_arg(arg: str, line: str) -> str:
    tmp = line.split(arg + "=")[1]
    for i, c in enumerate(tmp[1:]):
        if c == "\"":
            return tmp[1:i]
    raise ValueError(f"Argument {arg} not found in line: {line}")

def change_titles(path: str):
    with open(path, "r") as f:
        data = f.read()
    new_data = ""
    for line in data.splitlines():
        if "no title" in line and "href" in line:
            title = get_arg("href", line).split(".")[-2]
            line = line.replace("&lt;no title&gt;", title)
        elif "no title" in line:
            title = path.split(".")[-2]
            line = line.replace("&lt;no title&gt;", title)
        new_data += line + "\n"
    with open(path, "w") as f:
        f.write(new_data)

def replace_titles(app, exception):
    import os

    outdir = app.builder.outdir  # e.g. .../_build/html
    html_dir = os.path.join(outdir, "Trainsum", "methods")

    if not os.path.isdir(html_dir):
        app.logger.warning("replace_titles: directory %s does not exist", html_dir)
        return
    
    for file in os.listdir(html_dir):
        change_titles(os.path.join(html_dir, file))

def setup(app):
    app.connect('build-finished', replace_titles)
