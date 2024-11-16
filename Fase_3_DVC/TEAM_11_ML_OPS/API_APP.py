from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()

def run_command(command: str) -> str:
    """Run a shell command and return its output or raise an exception if it fails."""
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Fallo en el comamndo failed: {e.stderr}")

@app.post("/correr_pipeline")
def correr_pipeline():
    """
    Endpoint to run the pipeline using `dvc repro -f`.
    """
    output = run_command("dvc repro -f")
    return {
        "message": f"Pipeline Corrido Exitosamente" + "\n",
        "output": output
    }



@app.post("/almacenar_pipeline")
def store_data():
    """
    Endpoint to push data to the DVC remote using `dvc push`.
    """
    output = run_command("dvc push")
    return {"message": "Data almacenada correctamente",             
            "output": output}
