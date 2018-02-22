Getting started
===============

Prerequisites
-------------
This package is developed using Python 3.6. To generate documentation we used
Sphinx package, version 1.5.

Installation
------------
In order to install this package to your computer:

#. Download `Algorithm Predictor <https://git.zib.de/grips2017_synlab/Algorithm-Selection>`_ package.
#. From terminal open Python interactive shell
#. Import Python ``sys`` module and add path to the package to Python search path.

.. code::

  import sys
  sys.path.append('/path/to/package/root/directory/including/root/directory')

Next, you should install all packages this package depends on. To do that, navigate
to the project root directory and install all packages from requirements file. ::

  pip install -r requirements.txt

If you want to rebuild package's documentation, you will need to install Sphinx
package following instructions from `here <http://www.sphinx-doc.org/en/stable/tutorial.html>`_.

If you are not interesting in changing package's documentation, you can skip next
section and move directly to :ref:`tutorials-ref` section.

Documentation
-------------
Current documentation style is set to classic style. If you wish to change the style
you can choose between 13 different styles: `agogo`, `basic`, `bizstyle`, `classic`, `default`,
`epub`, `haiku`, `nature`, `nonav`, `pyramid`, `scrolls`, `sphinxdoc` and `traditional`.
In order to change between previously mentioned styles, it is enough to change
``html_theme`` property of documentation configuration file
(``package_root/docs/source/conf.py``) to some of the style names.

After you add some documentation or change something in documentation configuration
file, you should rebuild the documentation by running:

.. code:: shell

  sphinx-build -b html docs/source docs/build

This will generate documentation in HTML format. For building other documentation
formats, see `Sphinx documentation <http://www.sphinx-doc.org/en/stable/man/sphinx-build.html>`_.
In order to see documentation you generated, navigate to the ``docs/build``
subdirectory of the project root directory and open ``index.html`` in your browser.

If everything installed correctly, you are ready to move to the next section.
