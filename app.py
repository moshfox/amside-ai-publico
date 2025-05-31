import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests

# Cargar variables de entorno desde el archivo .env
# Esto cargará HUGGINGFACE_API_TOKEN y MODEL_URL de tu archivo .env
load_dotenv()

# Inicializa la aplicación Flask
# Le indicamos a Flask dónde buscar los archivos estáticos (tu index.html)
app = Flask(__name__, static_folder='static')

# Obtener la API Key y la URL del modelo de Hugging Face de las variables de entorno
# Es crucial que estas variables estén configuradas en tu entorno de despliegue (como Render)
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_URL = os.getenv("MODEL_URL")

# --- ¡¡¡VERIFICACIÓN CRÍTICA!!! ---
# Asegúrate de que las variables de entorno se hayan cargado correctamente.
# Si alguna no existe, la aplicación no se iniciará, lo cual es mejor que un error en tiempo de ejecución.
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("La variable de entorno HUGGINGFACE_API_TOKEN no está configurada. "
                     "Asegúrate de que tu archivo .env (local) o la configuración de entorno (en Render) la contengan.")
if not MODEL_URL:
    raise ValueError("La variable de entorno MODEL_URL no está configurada. "
                     "Asegúrate de que tu archivo .env (local) o la configuración de entorno (en Render) la contengan.")

# --- Ruta para servir el archivo HTML principal (tu frontend) ---
# Cuando alguien visite la URL raíz de tu backend (ej. https://amside-ai-backend.onrender.com/),
# Flask le enviará tu archivo index.html.
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# --- Ruta de la API para el chat (donde tu frontend enviará los mensajes) ---
@app.route('/api/chat', methods=['POST'])
def chat():
    # Obtener los datos JSON de la solicitud del frontend
    data = request.json
    messages = data.get('messages') # El frontend envía el historial completo de mensajes ChatML

    # Validación básica de la entrada
    if not messages:
        return jsonify({"error": "No se proporcionaron mensajes en la solicitud."}), 400

    # Mensaje del sistema para darle personalidad a tu IA
    # Este mensaje se insertará al inicio de la conversación enviada a Hugging Face
    system_message_content = (
        "Eres Amside AI, una inteligencia artificial creada por Hodelygil. "
        "Tu propósito principal es asistir en el estudio y el aprendizaje, "
        "proporcionando información y explicaciones detalladas. "
        "Sin embargo, también eres amigable y puedes mantener conversaciones informales y agradables. "
        "Responde de manera informativa y útil, pero con un tono conversacional."
    )

    # Construir el array de mensajes para la API de Hugging Face
    # Añadimos el mensaje del sistema al principio, seguido del historial de la conversación.
    # Esto asegura que el modelo siempre tenga contexto sobre su rol.
    hf_messages = [{"role": "system", "content": system_message_content}]
    for msg in messages:
        # Evitar añadir mensajes de sistema duplicados si por alguna razón ya venían del frontend
        if msg['role'] != 'system':
            hf_messages.append(msg)

    # Configuración del payload para la API de inferencia de Hugging Face
    payload = {
        "inputs": hf_messages, # Este es el historial de chat que el modelo usará
        "parameters": {
            "max_new_tokens": 500,       # Longitud máxima de la respuesta de la IA
            "do_sample": True,           # Habilita el muestreo (para respuestas más creativas)
            "temperature": 0.7,          # Controla la aleatoriedad (0.7 es un buen equilibrio)
            "top_p": 0.9,                # Muestreo con corte por probabilidad acumulada
            "repetition_penalty": 1.1,   # Penaliza la repetición de palabras/frases
            "return_full_text": False    # Importante: Solo queremos el texto generado por la IA, no el prompt completo
        }
    }

    # Configuración de los encabezados para la solicitud HTTP a Hugging Face
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {HUGGINGFACE_API_TOKEN}' # Aquí es donde se usa tu token seguro
    }

    try:
        # Realizar la solicitud POST a la API de inferencia de Hugging Face
        response = requests.post(MODEL_URL, headers=headers, json=payload)
        response.raise_for_status()  # Esto lanzará una excepción si la respuesta no es 200 OK (ej. 4xx o 5xx)

        hf_data = response.json()

        # Validar la estructura de la respuesta de Hugging Face
        if not hf_data or not isinstance(hf_data, list) or not hf_data[0].get('generated_text'):
            return jsonify({"error": "Respuesta inesperada de Hugging Face API.", "hf_response": hf_data}), 500

        # Extraer el texto generado por la IA
        ai_response_text = hf_data[0]['generated_text']

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

# --- Punto de entrada para ejecutar la aplicación Flask ---
if __name__ == '__main__':
    # Cuando ejecutas `python app.py`, esto se ejecuta.
    # `debug=True` es útil para el desarrollo local (recarga el servidor automáticamente, muestra errores detallados).
    # Para producción (como en Render), un servidor WSGI como Gunicorn se encargará de esto.
    app.run(debug=True, port=5000) # Inicia el servidor en localhost:5000