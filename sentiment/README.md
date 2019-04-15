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

Vistos estos resultados, se decidió contunuar sólamente agregando negaciones y cambiando el tokenizador,
sin filtrar stopwords.

### Búsqueda de parámetros con grid-search
Utilizando de sklearn.model\_selection ParameterGrid y fit\_grid\_point, se lograron encontrar parámetros
mejores que los existentes por defecto. Tanto para SVM como para LinearRegression se variaron los
parámetros "C" y "penalty". Para MNB se realizó la búsqueda sobre el parámetro "alpha".

### Inspección de modelos
Para la regresión logística entrenada con el corpus "ES", las features más negativas/positivas fueron
- N:
	> ! gracias buena : mejor ([-0.99926348 -0.54140291 -0.50538172 -0.49823128 -0.48591543])
	> mismo odio ni triste no ([0.47739055 0.60890999 0.65199539 0.80367345 0.97239018])
- NEU:
	> # gracias hoy ! ? ([-0.55027125 -0.54565408 -0.44861801 -0.39353667 -0.38358295])
	> NOT_pasa vez casa aunque sido ([0.33468925 0.35213721 0.37396061 0.3906679  0.410102  ])
- NONE:
	> no ... mal hoy ser ([-0.70112823 -0.49297417 -0.42632923 -0.34505574 -0.34179787])
	> vídeo jugar alguna semana ? ([0.39499268 0.39759467 0.39960258 0.45413998 0.99736069])
- P:
	> no triste ? ni odio ([-0.71764461 -0.50168295 -0.42048146 -0.39279131 -0.32097587])
	> mejor genial buen gracias ! ([0.63658058 0.6450265  0.85123413 0.85493357 1.14742911])

En general tiene sentido que palabras como "odio", "triste" sean asociadas con estados negativos,
así como "mejor", "genial", "buen" con estados positivos. Sin embargo, aparece como muy relevante
el signo de exclamación, incluso con la opción "Binary=True" en el CountVectorizer.


### Análisis de error
