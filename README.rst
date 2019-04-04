Procesamiento de Lenguaje Natural - FaMAF 2019
==============================================


Instalación
-----------

1. Se necesita el siguiente software:

   - Git
   - Pip
   - Python 3.4 o posterior
   - Virtualenvwrapper

   En un sistema basado en Debian (como Ubuntu), se puede hacer::

    sudo apt-get install git python-pip python3 virtualenvwrapper

2. Crear y activar un nuevo
   `virtualenv <http://virtualenv.readthedocs.org/en/latest/virtualenv.html>`_::

    mkvirtualenv --python=/usr/bin/python3 pln

3. Bajar el código::

    git clone https://github.com/PLN-FaMAF/PLN-2019.git

4. Instalarlo::

    cd PLN-2019
    pip install -r requirements.txt


Ejecución
---------

1. Activar el entorno virtual con::

    workon pln

2. Correr el script que uno quiera. Por ejemplo::

    python languagemodeling/scripts/train.py -h


Testing
-------

Correr nose::

    nosetests


Chequear Estilo de Código
-------------------------

Correr flake8 sobre el paquete o módulo que se desea chequear. Por ejemplo::

    flake8 languagemodeling

Correr Pylint de la misma manera. Por ejemplo::

    pylint languagemodeling


Jupyter (IDE y notebooks)
-------------------------

Correr Jupyter Lab (abre una ventana en el navegador por defecto)::

    jupyter lab


Resultados:
-----------

AddOneNGram:
    +----+-----------------+---------------+-----------+
    | n  | Log probability | Cross entropy | Perplexity|
    +====+=================+===============+===========+
    | 1  | -2530715        | 9.8453        | 920       |
    +----+-----------------+---------------+-----------+
    | 2  | -2886451        | 11.2292       | 2400      |
    +----+-----------------+---------------+-----------+
    | 3  | -3590102        | 13.9667       | 16009     |
    +----+-----------------+---------------+-----------+
    | 4  | -3827606        | 14.8906       | 30375     |
    +----+-----------------+---------------+-----------+

InterpolatedNGram:
    +----+-----------------+---------------+-----------+
    | n  | Log probability | Cross entropy | Perplexity|
    +====+=================+===============+===========+
    | 1  | -2530602        | 9.8448        | 919       |
    +----+-----------------+---------------+-----------+
    | 2  | -2007635        | 7.8103        | 224       |
    +----+-----------------+---------------+-----------+
    | 3  | -1896842        | 7.3793        | 166       |
    +----+-----------------+---------------+-----------+
    | 4  | -1861328        | 7.2411        | 151       |
    +----+-----------------+---------------+-----------+

BackOffNGram:
    +----+-----------------+---------------+-----------+
    | n  | Log probability | Cross entropy | Perplexity|
    +====+=================+===============+===========+
    | 1  | -2530602        | 9.8448        | 919       |
    +----+-----------------+---------------+-----------+
    | 2  | -1944316        | 7.5640        | 189       |
    +----+-----------------+---------------+-----------+
    | 3  | -1775654        | 6.9078        | 120       |
    +----+-----------------+---------------+-----------+
    | 4  | -1707795        | 6.6438        | 100       |
    +----+-----------------+---------------+-----------+
