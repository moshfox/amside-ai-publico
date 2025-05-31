import os
from flask import Flask, request, jsonify # Elimina send_from_directory
from dotenv import load_dotenv
import requests
from flask_cors import CORS

load_dotenv()

# Inicializa la aplicación Flask
# Ya no necesitamos especificar static_folder si no vamos a servir archivos estáticos desde Flask
app = Flask(__name__) 

# --- CONFIGURACIÓN DE CORS ---
CORS(app) # Sigue siendo necesario para que tu frontend (GitHub Pages) se comunique con esta API

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_URL = os.getenv("MODEL_URL")

if not HUGGINGFACE_API_TOKEN:
    raise ValueError("La variable de entorno HUGGINGFACE_API_TOKEN no está configurada.")
if not MODEL_URL:
    raise ValueError("La variable de entorno MODEL_URL no está configurada.")

# --- ¡¡¡IMPORTANTE: ELIMINAR ESTA RUTA!!! ---
# @app.route('/')
# def serve_index():
#     return send_from_directory(app.static_folder, 'index.html')
#
# Esta ruta ya NO es necesaria porque GitHub Pages servirá tu index.html.

# --- Ruta de la API para el chat (donde tu frontend enviará los mensajes) ---
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages')

    if not messages:
        return jsonify({"error": "No se proporcionaron mensajes en la solicitud."}), 400

    system_message_content = (
        "Eres Amside AI, una inteligencia artificial creada por Hodelygil. "
        "Tu propósito principal es asistir en el estudio y el aprendizaje, "
        "proporcionando información y explicaciones detalladas. "
        "Sin embargo, también eres amigable y puedes mantener conversaciones informales y agradables. "
        "Responde de manera informativa y útil, pero con un tono conversacional."
    )

    hf_messages = [{"role": "system", "content": system_message_content}]
    for msg in messages:
        if msg['role'] != 'system':
            hf_messages.append(msg)

    payload = {
        "inputs": hf_messages,
        "parameters": {
            "max_new_tokens": 500,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "return_full_text": False
        }
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {HUGGINGFACE_API_TOKEN}'
    }

    try:
        response = requests.post(MODEL_URL, headers=headers, json=payload)
        response.raise_for_status()

        hf_data = response.json()

        if not hf_data or not isinstance(hf_data, list) or not hf_data[0].get('generated_text'):
            return jsonify({"error": "Respuesta inesperada de Hugging Face API.", "hf_response": hf_data}), 500

        ai_response_text = hf_data[0]['generated_text']
        return jsonify({"response": ai_response_text})

    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con Hugging Face API: {e}")
        return jsonify({"error": f"Error al conectar con la IA: {e}. Por favor, inténtalo de nuevo más tarde."}), 500
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {e}. Por favor, contacta al soporte."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
