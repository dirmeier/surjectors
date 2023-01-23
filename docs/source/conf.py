from datetime import date

project = 'surjectors'
copyright = f"{date.today().year}, Simon Dirmeier"
author = 'Simon Dirmeier'
release = '0.1.3'

extensions = [

]

templates_path = ['_templates']
html_static_path = ['_static']

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": True,
    "exclude-members": "__repr__, __str__, __weakref__",
}

html_theme = "alabaster"


html_theme_options = {
    "repository_url": "https://github.com/dirmeier/surjectors",
    "use_repository_button": True,
    "use_download_button": False,
}

html_title = "surjectors"
