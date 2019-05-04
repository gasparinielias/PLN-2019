*_Aclaración: todos los resultados provienen de evaluar sobre el corpus de development,
  a excepción del ejercicio 6 (evaluado con test)_*

## Ejercicio 1
El archivo stats.py devolvió los siguientes datos:

| Corpus    | Tweets for training   | P     | N     | NEU   | NONE  |
|-----------|-----------------------|-------|-------|-------|-------|
| PE        | 1000                  | 231   | 242   | 166   | 361   |
| CR        | 800                   | 230   | 311   | 94    | 165   |
| ES        | 1008                  | 318   | 418   | 133   | 139   |


## Ejercicio 2
### Resultados eval.py:
Se evaluaron los distintos corpus utilizando distintos clasificadores (svm,
multinomial Naive Bayes y Logistic regression).

Los siguientes resultados son para modelos sin mejoras en el tokenizador o
modificacinoes en el preprocesamiento:


| Metrics           | PE       | ES       | CR       |
|-------------------|----------|----------|----------|
| *SVC*             |          |          |          |
| Accuracy          |  39.20%  |  51.58%  |  46.67%  |
| Macro-Precision   |  33.04%  |  39.53%  |  37.31%  |
| Macro-Recall      |  34.43%  |  38.75%  |  38.79%  |
| Macro-F1          |  33.72%  |  39.14%  |  38.04%  |
| *Regression*      |          |          |          |
| Accuracy          |  42.00%  |  53.16%  |  46.67%  |
| Macro-Precision   |  33.86%  |  37.13%  |  40.01%  |
| Macro-Recall      |  34.70%  |  37.34%  |  37.86%  |
| Macro-F1          |**34.28%**|  37.23%  |  38.91%  |
| *MultiNB*         |          |          |          |
| Accuracy          |  45.00%  |  54.55%  |  49.00%  |
| Macro-Precision   |  30.16%  |  54.19%  |  66.93%  |
| Macro-Recall      |  31.51%  |  34.96%  |  36.58%  |
| Macro-F1          |  30.82%  |**42.50%**|**47.31%**|


### Cambios en el vectorizer:
- Se reemplazó el tokenizador por defecto de sklearn por el word_tokenize de nltk,
el cual no elimina signos de puntuación ni emojis, los cuales pueden ser útiles
para determinar el sentimiento del tweet. Junto con ésto, se agregó manejo de negaciones,
intentando dar contexto a palabras como "no", "tampoco", "ni". Las negaciones ocurren
hasta encontrar un signo de puntuación o agotar un contador de negaciones max_neg. El
parámetro max_neg fue seleccionado empíricamente.
- Además, se filtraron las stop_words del español (nltk.corpus.stopwords) y se binarizaron
conteos de palabras.
- Por último se agregó un preprocesador, con la idea de modificar, eliminar o unificar
tokens, que de otra manera molestarían a la hora de clasificar.

Los resultados a continuación se obtuvieron clasificando con Multinomial Naive Bayes

| Metrics           | PE        | ES        | CR        |
|-------------------|-----------|-----------|-----------|
| *Prev results*    |           |           |           |
| Accuracy          | 45.00%    | 54.55%    | 49.00%    |
| Macro F1          | 30.82%    | 42.50%    | 47.31%    |
| *tokenizer y neg* |           |           |           |
| Accuracy          |**48.00%** |**56.72%** |**51.33%** |
| Macro F1          | 33.81%    |**43.91%** |**48.91%** |
| *stopwords y bin* |           |           |           |
| Accuracy          | 46.40%    | 55.93%    | 50.00%    |
| Macro F1          | **34.52%**| 31.94%    | 47.38%    |
| *preprocessor*    |           |           |           |
| Accuracy          | 45.40%    | 55.93%    | 49.67%    |
| Macro F1          | 34.42%    | 34.46%    | 44.12%    |


## Ejercicio 3: Búsqueda de parámetros con grid-search
Utilizando de sklearn.model\_selection ParameterGrid y fit\_grid\_point, se lograron encontrar parámetros
mejores que los existentes por defecto. Tanto para SVM como para LinearRegression se variaron los
parámetros "C" y "penalty".
Los mejores parámetros encontrados, con sus respectivos resultados, fueron:

| Corpus-clf | penalty | C      | Accuracy | Macro F1  |
|------------|---------|-------:|----------|-----------|
| PE-svm     | l1      | 0.1    | 49.60%   | 37.25%    |
| PE-maxent  | l2      | 1.0    | 47.40%   | 36.45%    |
| ES-svm     | l1      | 0.1    | 55.14%   | 46.52%    |
| ES-maxent  | l2      | 100.0  | 52.77%   | 42.68%    |
| CR-svm     | l1      | 0.1    | 44.00%   | 40.65%    |
| CR-maxent  | l2      | 1.0    | 53.00%   | 47.90%    |


## Ejercicio 4: Inspección de modelos
Para la regresión logística entrenada con el corpus "ES", las features más negativas/positivas fueron
- N:
	* ! gracias buena : mejor ([-0.99926348 -0.54140291 -0.50538172 -0.49823128 -0.48591543])

	* mismo odio ni triste no ([0.47739055 0.60890999 0.65199539 0.80367345 0.97239018])
- NEU:
	* \# gracias hoy ! ? ([-0.55027125 -0.54565408 -0.44861801 -0.39353667 -0.38358295])

	* NOT_pasa vez casa aunque sido ([0.33468925 0.35213721 0.37396061 0.3906679  0.410102  ])
- NONE:
	* no ... mal hoy ser ([-0.70112823 -0.49297417 -0.42632923 -0.34505574 -0.34179787])

	* vídeo jugar alguna semana ? ([0.39499268 0.39759467 0.39960258 0.45413998 0.99736069])
- P:
	* no triste ? ni odio ([-0.71764461 -0.50168295 -0.42048146 -0.39279131 -0.32097587])

	* mejor genial buen gracias ! ([0.63658058 0.6450265  0.85123413 0.85493357 1.14742911])

En general tiene sentido que palabras como "odio", "triste" sean asociadas con estados negativos,
así como "mejor", "genial", "buen" con estados positivos. Sin embargo, aparecen como muy relevantes
varios signos de puntuación, incluso con la opción "Binary=True" en el CountVectorizer.
Una posible solución sería filtrar los signos de puntuación que no sean emojis, para mantener la
idea de que los emojis tienen una carga significante de sentimiento.


## Ejercicio 5: Análisis de error
A partir del tweet a continuación, se realizaron modificaciones en los tokens que lo componen, con
el objetivo de ver las variaciones en las probabilidades que el clasificador asignó en cada caso:

Tweet original:
    Buen día.... Para delante una nueva semana con fe...!!!! Con fe ... Cuanto tiempo me aguantas con mi forma de ser...

Probabilidades iniciales:

|'N'    | 'NEU' | NONE' | 'P'   |
|-------|-------|-------|-------|
| 0.04173462 | 0.02647667  | 0.01663905 | 0.91514966 |

Para disminuir la diferencia de probabilidades de la clase predicha y la correcta, se reemplazaron palabras
que el clasificador considera "positivas" con palabras con carga "None" (enunciadas en el ejercicio anterior,
como ser: semana, alguna). Además, como era de esperar, modificaciones en las stopwords no afectaron a las
probabilidades, ya que el tokenizer se encarga de filtrarlas.

Tweet modificado:
    Otro día. Aunque para delante alguna semana con fe. Cuanto tiempo me aguantas con mi forma de ser.

Nuevas probabilidades para el tweet modificado (con y sin class\_weight balanced):

| Class weight  | 'N'           | 'NEU'         | 'NONE'        | 'P'        |
|---------------|---------------|---------------|---------------|------------|
|_Previous prob_| 0.04173462    | 0.02647667    | 0.01663905    | 0.91514966 |
| None          | 0.3165522     | 0.1470144     | 0.17219573    | 0.36423766 |
| Balanced      | 0.21668909    | 0.16951191    | 0.25684346    | 0.35695554 |

Como se puede ver, la probabilidad (sin class\_weight) de None aumentó, pero no lo suficiente.
Se vió que incluso una frase únicamente compuesta de palabras con mucho peso para "None", era clasificado como "None" con
apenas un 60% de posibilidades, y una frase vacía era clasificada con probabilidades muy desparejas.
Por ello se decidió utilizar class\_weight='balanced' en el clasificador. Con ésto, las probabilidades del string
vacío pasaron a ser:

| Class weight  | 'N'           | 'NEU'         | 'NONE'        | 'P'        |
|---------------|---------------|---------------|---------------|------------|
| None          | 0.36756399    | 0.17290057    | 0.18253403    | 0.27700141 |
| Balanced      | 0.29837558    | 0.24831629    | 0.23030293    | 0.22300519 |


## Ejercicio 6: Evaluación final
El resultado final medido sobre el conjunto de test para el corpus ES fue:

| Classifier    | Metric        | P      | N      | NEU       | NONE      |
|---------------|---------------|--------|--------|-----------|-----------|
| Maxent        | Presicion     | 56.09% | 56.58% | 13.33%    | 28.57%    |
| Maxent        | Recall        | 55.92% | 62.19% | 15.74%    | 16.79%    |
| SVM           | Presicion     | 60.10% | 50.73% | 100%      | 64.71%    |
| SVM           | Recall        | 54.21% | 86.18% | 0%        |  4.01%    |

| Classifier    | Metric        | ES     |
|---------------|---------------|--------|
| Maxent        | Accuracy      | 48.24% |
| Maxent        | Macro-F1      | 38.15% |
| SVM           | Accuracy      | 53.71% |
| SVM           | Macro-F1      | 47.37% |


## Ejercicio 7: Word embeddings
Se utilizó FastText para obtener word embeddings de las palabras de los tweets. Por motivos de recursos,
este ejercicio se realizó utilizando Google Colaboratory:
[Embeddings colab](https://colab.research.google.com/drive/1TU3o5J_AMzBK_vptIPyF9_6CE1v0D_Aj).
El nuevo conjunto de features fue el resultado de concatenar las obtenidas con el BoW y
la media (numpy.average) de las words embeddings de las palabras del tweet.


## Ejercicio 8: Lexicón de sentimientos
Se implementó una nueva clase para contar cantidad de palabras con carga positiva, neutra y negativa en los tweets.
Estos números, escalados con StandardScaler, fueron agregados como features al pipeline ya existente.
El lexicón utilizado fue [Spanish DAL](http://habla.dc.uba.ar/gravano/sdal.php?lang=esp). Dado que el mismo provee
un valor de "agrado" por cada palabra, en una escala del 1.0 al 3.0, los tokens de los tweets fueron clasificados
con polaridad de acuerdo a dos thresholds (low y high). Los mismos pueden buscarse con grid-search, como el resto
de los parámetros de los clasificadores.
