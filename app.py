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

# Define el mensaje del sistema fuera de la ruta para que sea fácilmente accesible
system_message_content = (
    "Eres Amside AI, una inteligencia artificial creada por Hodelygil. "
    "Tu propósito principal es asistir en el estudio y el aprendizaje, "
    "proporcionando información y explicaciones detalladas. "
    "Sin embargo, también eres amigable y puedes mantener conversaciones informales y agradables. "
    "Responde de manera informativa y útil, pero con un tono conversacional."
)

@app.route('/')
def health_check():
    return "¡Servidor de Amside AI funcionando correctamente!"
    data = request.json
    messages = data.get('messages')

    if not messages:
        return jsonify({"error": "No se proporcionaron mensajes en la solicitud."}), 400

    # --- MODIFICACIÓN CLAVE AQUÍ: Construir un único string para 'inputs' ---

    # Extraer el contenido del último mensaje del usuario
    last_user_message_content = ""
    for msg in reversed(messages): # Iterar en reversa para encontrar el último mensaje de usuario
        if msg['role'] == 'user':
            last_user_message_content = msg['content']
            break

    # Construir el prompt final como un único string, siguiendo el formato de chat de Zephyr
    # Esto incluye el mensaje del sistema y el último mensaje del usuario.
    prompt_string = (
        f"<s>[INST] <<SYS>>\n{system_message_content}\n<</SYS>>\n\n"
        f"{last_user_message_content} [/INST]"
    )

    payload = {
        "inputs": prompt_string, # Ahora 'inputs' es un único string, como lo exige el error 422
        "parameters": {
            "max_new_tokens": 500,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "return_full_text": False # Pedimos solo el texto generado nuevo
        }
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {HUGGINGFACE_API_TOKEN}'
    }

    try:
        response = requests.post(MODEL_URL, headers=headers, json=payload)
        response.raise_for_status() # Esto levantará un HTTPError para respuestas 4xx/5xx

        hf_data = response.json()

        if not hf_data or not isinstance(hf_data, list) or not hf_data[0].get('generated_text'):
            return jsonify({"error": "Respuesta inesperada de Hugging Face API.", "hf_response": hf_data}), 500

        ai_response_text = hf_data[0]['generated_text']

        # Zephyr, con return_full_text=False y el formato adecuado, debería devolver
        # solo la parte de la respuesta del asistente. Si aún incluye el prompt completo,
        # podríamos necesitar una lógica de extracción más robusta aquí.
        # Por ahora, si el texto generado empieza con el prompt, lo eliminamos.
        if ai_response_text.startswith(prompt_string):
             ai_response_text = ai_response_text[len(prompt_string):].strip()


        return jsonify({"response": ai_response_text})

    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            print(f"Hugging Face API Error Status: {e.response.status_code}")
            # Esto es lo que nos dio el error claro
            print(f"Hugging Face API Error Response Content: {e.response.text}")
        print(f"Error al conectar con Hugging Face API: {e}")
        return jsonify({"error": f"Error al conectar con la IA: {e}. Por favor, inténtalo de nuevo más tarde."}), 500
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {e}. Por favor, contacta al soporte."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
