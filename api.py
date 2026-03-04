from fastapi import FastAPI
from pydantic import BaseModel

# 1. Instanciamos la aplicación (El "Mesero")
app = FastAPI(title="API Industrial VAE", version="1.0")

# 2. Definimos el "Menú" (La forma de los datos usando Pydantic)
# Todo lo que nos envíen debe tener esta estructura exacta
class DatosSensor(BaseModel):
    id_pieza: str
    temperatura: float
    voltaje: float

# 3. Ruta GET (Consultar): Para ver si el servidor está vivo 
@app.get("/")
def inicio():
    return {"mensaje": "Servidor de IA en línea y esperando piezas."}

# 4. Ruta POST (Crear/Enviar): Para mandar datos y recibir un diagnóstico 
@app.post("/diagnosticar")
def diagnosticar_pieza(datos: DatosSensor):
    # Aquí en el futuro conectaremos tu VAE. 
    # Por ahora, simulamos una lógica simple:
    es_anomalo = datos.temperatura > 80.0 or datos.voltaje < 3.0
    
    # Respondemos siempre en formato JSON (Clave: Valor) 
    return {
        "id_pieza": datos.id_pieza,
        "estado": "Defectuoso" if es_anomalo else "Sano",
        "accion": "Descartar" if es_anomalo else "Aprobar"
    }