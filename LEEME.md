# Qué cambió, y qué hay que recorrer

Solo cambió `experiments/exp08_assimilation.py`. Los demás experimentos están
igual, así que **no hay que recorrer nada más**: exp01 a exp07, exp09, exp10 y
exp11 siguen valiendo.

## Los dos defectos que se arreglaron

**1. El conjunto se sorteaba de la climatología.**

Eso le daba una dispersión de unos 310 m en la anomalía de altura, y en este
modelo un error de pronóstico a seis horas es tres órdenes de magnitud menor
(se midió: una perturbación de 1.9 m sigue siendo de 1.9 m seis horas
después). Los dos filtros respondían con incrementos de 223 m, del tamaño de
la señal. El análisis ajustaba bien la altura, unos 3 m, pero los incrementos
de viento salían de covarianzas cruzadas estimadas con 40 miembros
climatológicos, así que el estado quedaba con la masa casi correcta y los
vientos equivocados, fuera de balance geostrófico, con picos de 88 a 91 m/s
contra una verdad de 80. Seis horas después el error volvía a 161 m, y la
dispersión del análisis había colapsado a 0.9 m frente a ese error.

No era culpa de ningún filtro. A los dos les entraba el mismo fondo
imposible, y por eso un solo cambio arregla los dos.

Ahora los miembros son la verdad desplazada por **diferencias escaladas** de
estados del registro, que son perturbaciones dinámicamente consistentes, con
la dispersión puesta en unos 19 m. El fondo se centra en un sorteo aparte y
no en la verdad: centrado en la verdad, la media del conjunto sale buena por
construcción y el conjunto aparenta estar seis veces sobredisperso, que es un
artefacto del montaje. Medido sobre los primeros diez ciclos, la razón entre
dispersión y error queda entre 0.86 y 1.07, el error de fondo baja de 23.6 a
12.9 m, el análisis se sostiene en 2.9 m y el viento máximo se queda cerca de
77 m/s. Los primeros `SPINUP_CYCLES` ciclos se corren sin puntuar.

**2. `cross=True` no hacía absolutamente nada.**

`AnalysisLETKF` y el filtro Cholesky local llaman a `model.get_ngb(i, r)` sin
argumento `cross`, y sus docstrings dicen que el modelo resuelve el despacho.
Pero `SWEModel.get_ngb` tiene `cross=False` por defecto y `SWEModel.__init__`
no acepta ninguna opción que lo cambie, así que el defecto siempre gana. El
`cross=True` que se le pasaba a `AnalysisFactory` caía en `**kwargs` y nunca
se leía.

O sea las 18 configuraciones con `cross=True` de la corrida anterior eran
idénticas a las 18 univariadas, hasta el último dígito del RMSE, y la columna
`incr_wind` era cero en todas las filas, incluidas las que existían para
mostrar que no lo era.

Ahora se expresa a través del modelo, con la subclase `CrossLocalized`. Con
radio 2 el dominio local pasa de 25 índices a 75, que es lo que permite que
una observación de altura alcance al viento. Verificado: con `cross=0` el
viento queda exactamente intacto (`rmse_wind_a == rmse_wind_b`, `incr_wind`
idénticamente cero), y con `cross=1` el incremento es distinto de cero.

Cuesta 1.9 veces más en LETKF y 1.3 veces más en EnKF-MC.

## Columna nueva

`spread_h_b`, la dispersión del fondo al lado de su error, para que la
calibración del conjunto quede en el CSV y no solo en la prosa. Una razón
cerca de uno es el objetivo; la versión anterior daba 0.006.

## Qué correr

Las **36 configuraciones completas**, no solo las que fallaron. Las 22 de
EnKF-MC que terminaron tienen el mismo defecto de origen y las 18 de
`cross=True` eran duplicados.

```bash
cd /mnt/data/swesphere-gmd
rm -f results/exp08_assimilation.csv
GIT_COMMIT=$(git rev-parse --short HEAD) python experiments/exp08_assimilation.py
```

Con sharding, igual que antes:

```bash
for i in 0 1 2 3; do
  SHARD_INDEX=$i SHARD_COUNT=4 GIT_COMMIT=$(git rev-parse --short HEAD) \
    nohup python experiments/exp08_assimilation.py \
    > logs/exp08_shard$i.log 2>&1 &
done
```

Necesita los `record_two_jets_seed*_L32.npy` que escribió exp04, que ya los
tienes.

---

# exp12, nuevo: cuánto crece un error pequeño

Salió de la depuración de exp08. Habíamos supuesto, sin medirlo, que una
ventana de seis horas era corta frente al crecimiento del error de este
flujo, por analogía con la práctica operacional. Es falso: una perturbación
de 1.9 m en la altura seguía siendo de 1.9 m seis horas después.

Esa suposición es lo primero que va a heredar quien use el banco de pruebas,
así que la curva tiene que estar en el artículo, y tiene que salir de código
archivado y no de un script suelto.

La medición no usa filtro. Dos trayectorias que difieren por una perturbación
pequeña se propagan lado a lado y se registra la distancia contra el tiempo
de pronóstico. La perturbación es una diferencia escalada de dos estados del
registro de exp04, o sea una diferencia de dos soluciones de las mismas
ecuaciones, balanceada por construcción. Se miden tres amplitudes iniciales,
porque el crecimiento solo es exponencial mientras el error es pequeño y la
amplitud a la que deja de serlo también vale la pena reportarla.

Cuesta minutos, no horas. Lee los mismos `record_*.npy` que exp08, ya está en
`docker-compose.yml` y en `run_all.sh` dentro de la etapa B, al lado de
exp08.

Suelto:

```bash
cd /mnt/data/swesphere-gmd
GIT_COMMIT=$(git rev-parse --short HEAD) python experiments/exp12_predictability.py
```

---

# exp13, nuevo: el estencil polar no era la causa

La meseta de 4e-3 del barrido a alfa = 90 sin esponja ya tenía dos candidatos
probados y descartados, el piso de |cos phi| y el protocolo del filtro. El
tercero era el par de diferencias unilaterales de **primer** orden en las
filas contiguas a cada polo, y era el que esperábamos que fuera la respuesta,
porque un término de primer orden dentro de un esquema de segundo orden es la
explicación de manual para un orden que no da dos.

No lo es. Reemplazándolas por el estencil unilateral de segundo orden:

```
  L      primer orden   segundo orden   razón
  16     2.882e-02      2.326e-02       0.81
  32     9.334e-03      8.475e-03       0.91
  64     4.684e-03      4.464e-03       0.95
  96     4.038e-03      3.928e-03       0.97
  orden observado  1.11           1.01
```

La forma del resultado lo dice más claro que el tamaño: la mejora se encoge
monótonamente de 19% a 3% y la razón converge a uno, o sea los dos estencils
comparten la misma meseta. Y el orden ajustado no mejora, baja de 1.11 a
1.01. Un término de primer orden dominante se comportaría al revés.

El primer bloque del experimento reproduce `exp01_convergence.csv` con el
estencil original, con acuerdo mejor que 1%. Eso no es ceremonia: comparar
una corrida parcheada contra un número publicado solo es una comparación si
la corrida sin parchear reproduce primero ese número.

Está en etapa A de `run_all.sh`, no depende de ningún registro. Cuesta unos
20 minutos, casi todos en L = 96.

Suelto:

```bash
cd /mnt/data/swesphere-gmd
GIT_COMMIT=$(git rev-parse --short HEAD) python experiments/exp13_polar_stencil.py
```

No toca el paquete. El estencil alternativo vive aquí, en el repositorio de
experimentos, porque se probó y se descartó, y una opción publicada existe
para usarse.
