# team_11_ml_ops

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

mlops MNA

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         team_11_ml_ops_module and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── team_11_ml_ops_module   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes team_11_ml_ops_module a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------




---

# Pendientes

- [ ] Referenciar a repositorios de GitHub mostrando actividad de parte de todos los integrantes del equipo.
- [ ] Entregar presentación ejecutiva de esta fase en PDF.
- [ ] Grabar video en equipo (5-10 min) con explicación del trabajo realizado en la fase, incluyendo el link de este video en el documento PDF.

# **Pendientes lista** 

## **Manipulación y Preparación de Datos** <- Juan
- [ ] **Importar los conjuntos de datos** con los que se trabajará.  
- [ ] **Limpiar los datos** eliminando valores nulos o inconsistentes.  
- [ ] **Transformar los datos** para prepararlos para análisis y modelado.  

## **Exploración y Preprocesamiento de Datos** <- Santi
- [ ] Realizar **análisis exploratorio de datos (EDA)** utilizando visualizaciones y estadísticas descriptivas.  
- [ ] **Preprocesar datos** aplicando normalización, codificación y/o reducción de dimensionalidad.  

## **Versionado de Datos** <- Juan
- [ ] Utilizar **herramientas de versionado de datos** para mantener un registro de cambios.  
- [ ] **Documentar las modificaciones** realizadas en los datos.  

## **Construcción, Ajuste y Evaluación de Modelos de Machine Learning** <- Santi
- [ ] **Seleccionar algoritmos** de Machine Learning adecuados para el problema.  
- [ ] **Entrenar los modelos** con los datos preprocesados.  
- [ ] Realizar **ajustes de hiperparámetros** para optimizar el rendimiento.  
- [ ] **Evaluar los modelos** utilizando métricas de rendimiento específicas.  

## **Aplicación de Mejores Prácticas en el Pipeline de Modelado** <- Otto
- [ ] Implementar **pipelines automáticos** para preprocesamiento, entrenamiento y evaluación.  
- [ ] **Documentar cada paso** del pipeline para garantizar reproducibilidad.  

## **Estructuración y Refactorización del Código**
- [ ] **Organizar el código** en módulos y funciones bien definidas.  
- [ ] Aplicar **principios de programación orientada a objetos (OOP)**.  
- [ ] **Refactorizar el código** para mejorar eficiencia y mantenibilidad.  
