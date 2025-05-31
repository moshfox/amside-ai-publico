import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

# Inicializa la aplicación Flask
app = Flask(__name__)
# Habilita CORS para permitir que tu frontend (ej. desde GitHub Pages)
# pueda hacer solicitudes a este backend.
CORS(app)

# Obtiene el token de Hugging Face y la URL del modelo de las variables de entorno.
# ¡Render inyectará estas variables cuando despliegues!
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# La MODEL_URL para Zephyr-7b-beta es la URL de su API de inferencia.
# Asegúrate de que esta variable de entorno esté configurada en Render.
MODEL_URL = os.getenv("MODEL_URL", "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta")

# --- ¡¡¡VERIFICACIÓN CRÍTICA!!! ---
# Asegúrate de que la API Key se haya cargado correctamente.
# Si no existe, la aplicación no se iniciará, lo cual es mejor que un error en tiempo de ejecución.
if not HF_API_TOKEN:
    raise ValueError("Error: La variable de entorno 'HF_API_TOKEN' no está configurada. "
                     "Asegúrate de definirla en Render.")
if not MODEL_URL:
    raise ValueError("Error: La variable de entorno 'MODEL_URL' no está configurada. "
                     "Asegúrate de definirla en Render (debería ser la URL de la API de inferencia del modelo).")

# Cabeceras para la solicitud HTTP, incluyendo la autorización con tu token
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- Definición de la Personalidad de la IA ---
# Este es el mensaje del sistema que se enviará al modelo para darle contexto sobre su rol.
SYSTEM_MESSAGE_CONTENT = (
    "Eres Amside AI, una inteligencia artificial creada por Hodelygil. "
    "Tu propósito principal es asistir en el estudio y el aprendizaje, "
    "proporcionando información y explicaciones detalladas. "
    "Sin embargo, también eres amigable y puedes mantener conversaciones informales y agradables. "
    "Responde de manera informativa y útil, pero con un tono conversacional y cercano."
)

def query_huggingface_model(payload):
    """
    Función auxiliar para enviar la solicitud a la API de Hugging Face.
    """
    response = requests.post(MODEL_URL, headers=HEADERS, json=payload)
    response.raise_for_status()  # Lanza una excepción para respuestas HTTP 4xx/5xx (errores)
    return response.json()

@app.route('/generate', methods=['POST'])
def generate_text():
    """
    Endpoint principal '/generate' que recibe el historial de mensajes del frontend
    y devuelve la respuesta de la IA.
    """
    data = request.get_json()
    # El frontend ahora envía un array de mensajes (historial de chat)
    messages_from_frontend = data.get('messages', [])

    # Validación básica de la entrada
    if not messages_from_frontend:
        return jsonify({"error": "No se proporcionaron mensajes en la solicitud."}), 400

    # Construir el array de mensajes para la API de Hugging Face
    # Añadimos el mensaje del sistema al principio para definir la personalidad.
    # Luego, añadimos el historial de la conversación del frontend.
    hf_messages = [{"role": "system", "content": SYSTEM_MESSAGE_CONTENT}]
    for msg in messages_from_frontend:
        # Asegurarse de que no estamos duplicando el mensaje de sistema si el frontend ya lo envía
        if msg.get('role') != 'system':
            hf_messages.append(msg)

    # Configuración del payload para la API de inferencia de Hugging Face
payload = {
    "inputs": hf_messages,
    "parameters": {
        "max_new_tokens": 500,
        "temperature": 0.7,
        # "do_sample": True,          # <-- Try commenting these out
        # "top_p": 0.95,              # <-- Try commenting these out
        # "repetition_penalty": 1.2   # <-- Try commenting these out
    }
}
    try:
        # Realizar la solicitud POST a la API de inferencia de Hugging Face
        response = requests.post(MODEL_URL, headers=HEADERS, json=payload)
        response.raise_for_status()  # Esto lanzará una excepción si la respuesta no es 200 OK

        hf_data = response.json()

        # Validar la estructura de la respuesta de Hugging Face
        # Para modelos de chat, la respuesta suele ser un array con un diccionario que contiene 'generated_text'
        if not hf_data or not isinstance(hf_data, list) or not hf_data[0].get('generated_text'):
            print(f"Respuesta inesperada de Hugging Face: {hf_data}")
            return jsonify({"error": "Respuesta inesperada de Hugging Face API.", "hf_response": hf_data}), 500

        # Extraer el texto generado por la IA
        ai_response_text = hf_data[0]['generated_text']

        # Limpieza adicional: algunos modelos pueden incluir las etiquetas de turno o el prompt
        # Asegurarse de que solo se devuelva la respuesta del asistente.
        ai_response_text = ai_response_text.split('</s>')[0].strip() # Eliminar cualquier terminador de conversación
        if ai_response_text.startswith('<|assistant|>'):
            ai_response_text = ai_response_text.replace('<|assistant|>', '').strip()
        if ai_response_text.startswith('<|user|>'): # En caso de que el modelo "imite" al usuario
            ai_response_text = ai_response_text.replace('<|user|>', '').strip()
        if ai_response_text.startswith('<|system|>'): # En caso de que el modelo "imite" al sistema
            ai_response_text = ai_response_text.replace('<|system|>', '').strip()


        # Devolver la respuesta de la IA al frontend en formato JSON
        return jsonify({"response": ai_response_text})

    except requests.exceptions.RequestException as e:
        # Captura errores relacionados con la conexión HTTP (red, DNS, etc.)
        print(f"Error al conectar con Hugging Face API: {e}")
        return jsonify({"error": f"Error al conectar con la IA: {e}. Por favor, inténtalo de nuevo más tarde."}), 500
    except Exception as e:
        # Captura cualquier otro error inesperado en el servidor
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {e}. Por favor, contacta al soporte."}), 500

# Punto de entrada para ejecutar la aplicación Flask
if __name__ == '__main__':
    # Cuando ejecutas `python app.py`, esto se ejecuta.
    # `debug=True` es útil para el desarrollo local (recarga el servidor automáticamente, muestra errores detallados).
    # Para producción (como en Render), un servidor WSGI como Gunicorn se encargará de esto.
    # Render proporcionará el puerto a través de la variable de entorno 'PORT'.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
