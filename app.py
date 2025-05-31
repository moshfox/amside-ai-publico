import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
# Habilita CORS para permitir que tu frontend (ej. desde GitHub Pages)
# pueda hacer solicitudes a este backend.
CORS(app)

# Obtiene el token de Hugging Face de las variables de entorno.
# ¡Render inyectará esta variable cuando despliegues!
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# URL del modelo Zephyr-7b-beta en la API de inferencia de Hugging Face
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

# Cabeceras para la solicitud HTTP, incluyendo la autorización con tu token
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Plantilla de prompt para el modelo Zephyr. Es crucial usar el formato ChatML
# para que el modelo entienda el rol del usuario y del asistente.
PROMPT_TEMPLATE = """<|user|>
{}</s>
<|assistant|>
"""

def query_huggingface_model(payload):
    """
    Función auxiliar para enviar la solicitud a la API de Hugging Face.
    """
    if not HF_API_TOKEN:
        # Lanza un error si el token no está configurado (importante para depuración)
        raise ValueError("HF_API_TOKEN no está configurado. No se puede realizar la inferencia.")

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()  # Lanza una excepción para respuestas HTTP 4xx/5xx (errores)
    return response.json()

@app.route('/generate', methods=['POST'])
def generate_text():
    """
    Endpoint principal '/generate' que recibe el mensaje del usuario y devuelve la respuesta de la IA.
    """
    data = request.get_json()
    user_message = data.get('message')

    # Valida que se haya enviado un mensaje
    if not user_message:
        return jsonify({"error": "No se proporcionó ningún mensaje"}), 400

    # Formatea el mensaje del usuario según la plantilla del modelo Zephyr
    formatted_prompt = PROMPT_TEMPLATE.format(user_message)

    try:
        # Prepara el 'payload' (cuerpo de la solicitud) para la API de Hugging Face
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 200,      # Número máximo de tokens que la IA puede generar
                "temperature": 0.7,         # Controla la aleatoriedad de la respuesta (0.0 = determinista, 1.0 = muy creativo)
                "top_p": 0.9,               # Muestreo Top-P, ayuda a la diversidad sin salirse del tema
                "do_sample": True,          # Activa el muestreo para respuestas menos predecibles
                "return_full_text": False   # Importante: solo devuelve la parte generada por el asistente
            }
        }
        output = query_huggingface_model(payload)

        # Procesa la respuesta de Hugging Face
        if output and isinstance(output, list) and 'generated_text' in output[0]:
            ai_response = output[0]['generated_text'].strip()

            # Limpieza adicional: a veces el modelo puede incluir etiquetas o el prompt de entrada
            ai_response = ai_response.split('</s>')[0].strip() # Elimina cualquier terminador de conversación
            if ai_response.startswith('<|assistant|>'):
                ai_response = ai_response.replace('<|assistant|>', '').strip()
            if ai_response.startswith('<|user|>'):
                ai_response = ai_response.replace('<|user|>', '').strip()

            return jsonify({"response": ai_response})
        else:
            return jsonify({"error": "Respuesta inesperada de la IA"}), 500

    except ValueError as e:
        # Maneja el caso en que el token no esté configurado
        return jsonify({"error": str(e)}), 500
    except requests.exceptions.RequestException as e:
        # Captura errores de conexión o de la API de Hugging Face
        print(f"Error al conectar con la API de Hugging Face: {e}")
        return jsonify({"error": f"Error al conectar con la IA: {e}"}), 500
    except Exception as e:
        # Captura cualquier otro error inesperado en el servidor
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500

# Punto de entrada para el servidor web.
# Render proporcionará el puerto a través de la variable de entorno 'PORT'.
# Si se ejecuta localmente sin 'PORT', usará el 5000 por defecto.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
