import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
from flask_cors import CORS # Para permitir solicitudes desde tu frontend en GitHub Pages

# Cargar variables de entorno desde .env
load_dotenv()

app = Flask(__name__)
CORS(app) # Habilitar CORS para todas las rutas

# Obtener el token de la API de Hugging Face desde las variables de entorno
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Plantilla de prompt para Zephyr (importante para que el modelo funcione bien)
# Zephyr usa el formato ChatML: <|system|>, <|user|>, <|assistant|>
PROMPT_TEMPLATE = """<|user|>
{}</s>
<|assistant|>
"""

def query_huggingface_model(payload):
    """
    Envía la solicitud al modelo de Hugging Face.
    """
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()  # Lanza una excepción si la solicitud no fue exitosa
    return response.json()

@app.route('/generate', methods=['POST'])
def generate_text():
    """
    Endpoint para generar texto con la IA.
    Recibe el mensaje del usuario y devuelve la respuesta de la IA.
    """
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "Mensaje no proporcionado"}), 400

    # Aplicar la plantilla de prompt al mensaje del usuario
    formatted_prompt = PROMPT_TEMPLATE.format(user_message)

    try:
        # Preparar el payload para Hugging Face
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 200,  # Máximo de tokens a generar
                "temperature": 0.7,     # Aleatoriedad de la respuesta (0.0 a 1.0)
                "top_p": 0.9,           # Muestreo de tokens (para diversidad)
                "do_sample": True,      # Activar muestreo
                "return_full_text": False # Solo devolver el texto generado por el asistente
            }
        }
        output = query_huggingface_model(payload)

        # La respuesta de Hugging Face para este modelo es una lista de diccionarios
        # y el texto generado estará en 'generated_text' del primer elemento.
        if output and isinstance(output, list) and 'generated_text' in output[0]:
            ai_response = output[0]['generated_text'].strip()
            # Opcional: limpiar la respuesta si contiene el prompt inicial o algo inesperado
            # Algunas veces, el modelo puede repetir partes del prompt aunque `return_full_text` sea False.
            # Puedes ajustar esto si observas comportamientos indeseados.
            if ai_response.startswith(user_message):
                ai_response = ai_response[len(user_message):].strip()
            if ai_response.startswith('<|assistant|>'): # Asegurarse de que no incluya la etiqueta del asistente
                ai_response = ai_response.replace('<|assistant|>', '').strip()
            if ai_response.startswith('<|user|>'): # Puede que el modelo "imite" la entrada del usuario
                ai_response = ai_response.replace('<|user|>', '').strip()


            # Asegurarse de que la respuesta no termine con la etiqueta de fin de usuario si el modelo la genera accidentalmente.
            ai_response = ai_response.split('</s>')[0].strip()


            return jsonify({"response": ai_response})
        else:
            return jsonify({"error": "Respuesta inesperada de la IA"}), 500

    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con la API de Hugging Face: {e}")
        return jsonify({"error": f"Error al conectar con la IA: {e}"}), 500
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500

# Punto de entrada para el servidor web
if __name__ == '__main__':
    # Para desarrollo local: app.run(debug=True)
    # Para Render, el puerto será proporcionado por el entorno, no lo establecemos aquí
    # Render usará gunicorn o similar para ejecutar la aplicación
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
