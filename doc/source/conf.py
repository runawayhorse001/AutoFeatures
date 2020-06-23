# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'AutoFeatures: PySpark Auto Feature Selector'
copyright = '2020, Wenqiang Feng'
author = 'Wenqiang Feng'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.graphviz',
              'sphinxcontrib.dd',
              'sphinx_tabs.tabs']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', 'images']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

version = '1.0'

# The full version, including alpha/beta/rc tags.
release = '1.0'

# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = 'images/icon.ico'

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Options for LaTeX output
# ------------------------

latex_elements = {
    # The paper size ('letter' or 'a4').
    #latex_paper_size = 'a4',

    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '11pt',

    # Additional stuff for the LaTeX preamble.
    #latex_preamble = '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class
# [howto/manual]).
latex_documents = [
  ('index', 'AutoFeatures.tex', 'AutoFeatures: PySpark Auto Feature Selector',
   'Wenqiang Feng', 'manual'),
]
# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = 'images/logo.png'
# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = 'images/snake_theta2-trans.png'
#latex_logo = 'images/theano_logo_allblue_200x46.png'

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_use_modindex = True