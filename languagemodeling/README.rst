Proyecto 1
==========

Ejercicio 1
-----------

El corpus elegido fue "lavoztexttodump", cuyo contenido es una recopilación de noticias del diario La Voz del Interior. Se realizaron pequeñas limpiezas, quitando caracteres no UTF-8 y encodings HTML (como &#266;).
Para tokenizar el corpus se utilizaron las funciones word_tokenize() y sent_tokenize(), del módulo nltk.tokenize.

Ejercicios 2 y 3
----------------

Se implementó una función _compute_counts() general para las clases que heredan de NGram (AddOne, Interpolated y Backoff). Es por ello que incluye un parámetro "all_ngrams" que, de ser True, computa counts para k-gramas con 0 <= k <= n.
Por otro lado, para el generador de oraciones se asume que la sumatoria de cond_prob(X, x1..xM) sobre X, tal que x1..xMX es visto en entrenamiento, es 1 (cosa que no ocurre cuando hay discount o addone). Dicho ésto, dado el comienzo de oración, se puede generar la palabra a continuación a partir de una muestra de una variable aleatoria con distribución uniforme(0,1) (método de la transformada inversa).

Ejercicio 4
-----------

Dado que la clase AddOneNGram hereda de NGram, la única función que fue necesario reimplementar fue la de probabilidad condicional, para agregar smoothing.

Ejercicio 5
-----------

Se implementaron funciones para obtener las métricas log prob, cross-entropy y perplexity dentro de la clase LanguageModel. Ésto fue posible dado que las métricas se calculan en función de la probabilidad condicional de cada modelo, independientemente de su implementación.

Ejercicios 6 y 7
----------------

Se implementaron dos nuevos métodos de smoothing, cada ejercicio correspondiéndose con las clases InterpolatedNGram y BackOffNGram.
En ambos casos se requiere el valor de un hiper parámetro (gamma y beta respectivamente) para calcular la probabilidad condicional. Para las ejecuciones en que dichos parámetros no son dados, se busca un valor cercano al óptimo por grid-search. El rángo de búsqueda se acotó empíricamente tras varias pruebas sobre el conjunto de development.


Resultados:
-----------
A continuación se reportan las métricas obtenidas para los distintos métodos de smoothing, para valores de n entre 1 y 4.

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
