## Ejercicio 1
El archivo stats.py devolvió los siguientes datos:

| Estadísticas básicas    | Cantidad |
|-------------------------|----------|
| Oraciones               | 17378    |
| Ocurrencias de palabras | 517194   |
| Palabras (únicas)       | 46501    |
| Tags                    | 85       |
 
### Etiquetas más frecuentes

| tag            | significado   | freq        | %        | top |
|----------------|---------------|-------------|----------|-----|
| sp000            | preprosición | 79884        | 15.45    | de, en, a, del, con|
| nc0s000        | nombre común singular |63452        | 12.27    | presidente, equipo, partido, país, año|
| da0000        | artículos | 54549        | 10.55    | la, el, los, las, El|
| aq0000        | adjetivo calificativo | 33906        | 6.56    | pasado, gran, mayor, nuevo, próximo|
| fc            | puntuación | 30147        | 5.83    | ,|
| np00000        | nombre propio | 29111        | 5.63    | Gobierno, España, PP, Barcelona, Madrid|
| nc0p000        | nombre común plural | 27736        | 5.36    | años, millones, personas, países, días|
| fp            | puntuación | 17512        | 3.39    | .|
| rg            | adverbio | 15336        | 2.97    | más, hoy, también, ayer, ya|
| cc            | conjunción | 15023        | 2.90    | y, pero, o, Pero, e|


### Nieveles de ambigüedad

| n    | words    | %        | top |
|------|----------|----------|-----|
| 1    | 43972    | 94.56    | (,, con, por, su, El) |
| 2    | 2318     | 4.98     | (el, en, y, ", los) |
| 3    | 180      | 0.39     | (de, la, ., un, no) |
| 4    | 23       | 0.05     | (que, a, dos, este, fue) |
| 5    | 5        | 0.01     | (mismo, cinco, medio, ocho, vista) |
| 6    | 3        | 0.01     | (una, como, uno) |
| 7    | 0        | 0.00     | () |
| 8    | 0        | 0.00     | () |
| 9    | 0        | 0.00     | () |


## Ejercicio 2
### Baseline tagger

El _BaselineTagger_ asigna a cada palabra la etiqueta más frecuente vista para dicha palabra (o nombre común singular en caso de ser una palabra desconocida). Con este algoritmo sencillo se obtuvieron los resultados:

Accuracy: 87.58%
Accuracy para palabras conocidas: 95.27%
Accuracy para palabras desconocidas: 18.01%

## Ejercicio 3
Para obtener la matriz de confusión correr **eval.py** con la opción -c. Output: confusion\_matrix.png.

## Ejercicio 5
Resultados de Maximum-likelihood Hidden Markov Models para distintos n. Se puede ver que el tiempo de evaluación crece exponencialmente con el tamaño del n-grama, aunque la performance (tanto para palabras conocidas como desconocidas) mejora significativamente entre n=1 y n=3.

| n         | Accuracy (gen/known/unk)  | Tiempo de evaluación (real/user/sys) |
|-----------|---------------------------|--------------------------------------|
| 1         | 85.84% / 95.28% / 0.45%   | 1m 48s / 1m 48s / 0,8s               |
| 2         | 91.34% / 97.63% / 34.33%  | 1m 53s / 1m 53s / 0,6s               |
| 3         | 91.87% / 97.65% / 39.50%  | 4m 22s / 4m 20s / 1,4s               |
| 4         | 91.61% / 97.31% / 40.01%  | 22m 18s / 22m 1s / 7,0s              |

## Ejercicio 6
### Three words classifier

La idea es, para cada palabra de cada oración del corpus, vectorizar un diccionario con la información de la palabra actual, la siguiente y la anterior, para luego entrenar un clasificador multiclase.

| Classifier            | Accuracy (gen/known/unk) | Tiempo de evaluación (real/user/sys)   |
|-----------------------|-----------|-----------------------|
| SVM                   | 94.11% / 97.57% / 62.76%  | 51,8s / 35,8s / 16,4s                 |
| Logistic Regression   | 91.69% / 95.01% / 61.67%  | 52,7s / 36,5s / 16,7s                 |
| Naive Bayes           | 84.28% / 88.07% / 49.99%  | 5m 21s / 3m 12s / 2m 8s               |
