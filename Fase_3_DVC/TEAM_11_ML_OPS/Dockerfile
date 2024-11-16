# Utiliza una imagen base de Python
FROM python:3.12-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos locales (como requirements.txt) al contenedor
COPY requirements.txt /app/

# Instala las dependencias desde el archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto que utilizará FastAPI o Jupyter
EXPOSE 8000

# Comando por defecto para ejecutar FastAPI con Uvicorn o JupyterLab (puedes cambiar según lo que desees)
CMD ["uvicorn", "app:main", "--reload"]
