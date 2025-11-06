Installation
============

Step 1: Clone the Repository
----------------------------

Using Git::

    git clone https://github.com/diedrichsenlab/AnatSearchlight.git
    cd AnatSearchlight

Or use `GitHub Desktop <https://desktop.github.com/>`_.

Step 2: Install Python (≥ 3.9)
------------------------------

This project requires **Python 3.9 or later**.

Option A: Using pyenv (Recommended on macOS/Linux)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install pyenv::

    brew update
    brew install pyenv

Configure your shell::

    echo 'if command -v pyenv 1>/dev/null 2>&1; then eval "$(pyenv init -)"; fi' >> ~/.bash_profile
    source ~/.bash_profile

Install Python::

    pyenv install 3.9.0
    pyenv global 3.9.0

Option B: Using system Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure that running::

    python --version

reports version 3.8 or higher.


Step 3: Install Dependencies
----------------------------

Make sure you have the latest version of ``pip`` and then install the required packages::

    pip install --upgrade pip
    pip install -r requirements.txt
