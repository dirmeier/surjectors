from datetime import date

project = "Surjectors"
copyright = f"{date.today().year}, the Surjectors developers"
author = "the Surjectors developers"

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    'sphinx_autodoc_typehints',
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_math_dollar",
    "IPython.sphinxext.ipython_console_highlighting",
    'sphinx_design',
]


templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ['theme.css']

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": True,
    "exclude-members": "__repr__, __str__, __weakref__",
}

exclude_patterns = [
    "_build",
    "build",
    "Thumbs.db",
    ".DS_Store",
    "notebooks/.ipynb_checkpoints",
    "examples/*ipynb",
    "examples/*py"
]

html_theme = "furo"

html_theme_options = {
    "repository_url": "https://github.com/dirmeier/surjectors",
    "use_repository_button": True,
    "use_download_button": False,
    "extra_navbar": ""
}

html_title = "Surjectors 🚀"