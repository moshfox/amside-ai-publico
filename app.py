import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_URL = os.getenv("MODEL_URL")

if not HUGGINGFACE_API_TOKEN:
    raise ValueError("La variable de entorno HUGGINGFACE_API_TOKEN no está configurada.")
if not MODEL_URL:
    raise ValueError("La variable de entorno MODEL_URL no está configurada.")

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages')

    if not messages:
        return jsonify({"error": "No se proporcionaron mensajes en la solicitud."}), 400

    # --- MODIFICACIÓN CLAVE AQUÍ ---
    # Vamos a construir los mensajes para Hugging Face.
    # Para Zephyr y modelos similares, a menudo la API de inferencia espera solo 'user'/'assistant' roles
    # en el array de 'inputs'. El mensaje del sistema puede ser ignorado o causar un 422 si no está formateado
    # de una manera específica por la API.
    # Por simplicidad y para solucionar el 422, quitaremos el rol 'system' explícito aquí,
    # y solo pasaremos los mensajes de 'user' y 'assistant' al modelo.

    hf_messages_for_api = []
    for msg in messages:
        if msg['role'] == 'user' or msg['role'] == 'assistant':
            hf_messages_for_api.append(msg)
        # Puedes añadir una lógica aquí si quieres incluir el system_message_content
        # como parte del primer mensaje de usuario, pero para el 422, simplifiquemos.

    payload = {
        "inputs": hf_messages_for_api, # Pasamos la lista filtrada de mensajes
        "parameters": {
            "max_new_tokens": 500,
            # Vamos a quitar temporalmente los otros parámetros para ver si alguno de ellos
            # está causando el 422. Si funciona, podemos añadirlos de nuevo uno a uno.
            # "do_sample": True,
            # "temperature": 0.7,
            # "top_p": 0.9,
            # "repetition_penalty": 1.1,
            # "return_full_text": False # La API de Zephyr suele funcionar con false
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

        # Si return_full_text es False, la respuesta debería ser solo el texto del asistente.
        # Si la API de Zephyr te devuelve la conversación completa (incluyendo tu input),
        # necesitarías una lógica aquí para extraer solo la última respuesta del asistente.
        # Pero, por ahora, devolvamos lo que venga.

        return jsonify({"response": ai_response_text})

    except requests.exceptions.RequestException as e:
        # Esto nos dará más detalle si el 422 trae un mensaje específico de Hugging Face
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 422:
            print(f"Hugging Face API 422 Error Response Content: {e.response.text}")
        print(f"Error al conectar con Hugging Face API: {e}")
        return jsonify({"error": f"Error al conectar con la IA: {e}. Por favor, inténtalo de nuevo más tarde."}), 500
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {e}. Por favor, contacta al soporte."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
