import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
from flask_cors import CORS # Importa Flask-CORS

# Carga las variables de entorno desde el archivo .env
load_dotenv()

app = Flask(__name__)
# Habilita CORS para permitir solicitudes desde tu frontend en GitHub Pages
CORS(app)

# Obtén las variables de entorno
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_URL = os.getenv("MODEL_URL")

# Verifica que las variables de entorno estén configuradas
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

# --- RUTAS DE TU SERVIDOR ---

# Ruta de verificación de estado (Health Check)
# Responde a las solicitudes GET a la raíz del servidor.
# Esto es útil para comprobar si el servidor está funcionando y para evitar errores 404
# cuando tu frontend o un servicio de monitoreo intente acceder a la raíz.
@app.route('/', methods=['GET'])
def health_check():
    return "¡Servidor de Amside AI funcionando correctamente! Dirígete a /api/chat para interactuar con la IA."

# Ruta principal de la API de chat
# Solo acepta solicitudes POST, como está configurado en tu frontend.
@app.route('/api/chat', methods=['POST'])
def chat():
    # Intenta obtener los datos JSON de la solicitud
    data = request.json
    messages = data.get('messages')

    # Valida que se hayan proporcionado mensajes
    if not messages:
        return jsonify({"error": "No se proporcionaron mensajes en la solicitud."}), 400

    # Extraer el contenido del último mensaje del usuario
    last_user_message_content = ""
    # Itera en reversa para encontrar el último mensaje de usuario de manera eficiente
    for msg in reversed(messages):
        if msg['role'] == 'user':
            last_user_message_content = msg['content']
            break

    # Si por alguna razón no hay un mensaje de usuario, manejar el caso
    if not last_user_message_content:
        return jsonify({"error": "No se encontró ningún mensaje de usuario válido en la solicitud."}), 400

    # Construir el prompt final como un único string, siguiendo el formato de chat de Zephyr.
    # Esto incluye el mensaje del sistema y el último mensaje del usuario, envueltos en las etiquetas de Zephyr.
    prompt_string = (
        f"<s>[INST] <<SYS>>\n{system_message_content}\n<</SYS>>\n\n"
        f"{last_user_message_content} [/INST]"
    )

    # Prepara el payload para la solicitud a la API de Hugging Face
    payload = {
        "inputs": prompt_string, # 'inputs' debe ser un único string para la mayoría de los modelos de texto
        "parameters": {
            "max_new_tokens": 500,        # Número máximo de tokens a generar
            "do_sample": True,            # Muestreo para mayor diversidad en las respuestas
            "temperature": 0.7,           # Controla la aleatoriedad (0.7 es un buen equilibrio)
            "top_p": 0.9,                 # Muestreo de núcleo (considera los tokens más probables)
            "repetition_penalty": 1.1,    # Penaliza la repetición de tokens
            "return_full_text": False     # Pide solo el texto generado nuevo, no el prompt completo
        }
    }

    # Prepara las cabeceras para la solicitud a Hugging Face
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {HUGGINGFACE_API_TOKEN}'
    }

    try:
        # Realiza la solicitud POST a la API de Hugging Face
        response = requests.post(MODEL_URL, headers=headers, json=payload)
        # Lanza una excepción HTTPError para respuestas 4xx o 5xx
        response.raise_for_status()

        # Parsea la respuesta JSON de Hugging Face
        hf_data = response.json()

        # Valida la estructura de la respuesta de Hugging Face
        if not hf_data or not isinstance(hf_data, list) or not hf_data[0].get('generated_text'):
            return jsonify({"error": "Respuesta inesperada de Hugging Face API.", "hf_response": hf_data}), 500

        # Extrae el texto generado por la IA
        ai_response_text = hf_data[0]['generated_text']

        # Opcional: Si el modelo devuelve el prompt completo a pesar de return_full_text=False,
        # puedes intentar eliminarlo de la respuesta. Esto es un ajuste fino si el modelo no se comporta como se espera.
        if ai_response_text.startswith(prompt_string):
            ai_response_text = ai_response_text[len(prompt_string):].strip()
        # Asegúrate de eliminar cualquier marcador de fin de turno del modelo (ej. "</s>")
        ai_response_text = ai_response_text.replace("</s>", "").strip()


        # Retorna la respuesta de la IA en formato JSON
        return jsonify({"response": ai_response_text})

    except requests.exceptions.HTTPError as e:
        # Maneja errores específicos de HTTP (4xx, 5xx) de la API de Hugging Face
        error_detail = e.response.text if e.response else "No hay detalles de respuesta."
        print(f"Hugging Face API HTTP Error Status: {e.response.status_code}")
        print(f"Hugging Face API Error Response Content: {error_detail}")
        return jsonify({
            "error": f"Error del servidor ({e.response.status_code}):\n{error_detail}",
            "details": f"Error al conectar con la IA de Hugging Face. Código de estado: {e.response.status_code}"
        }), e.response.status_code
    except requests.exceptions.ConnectionError as e:
        # Maneja errores de conexión (ej. el MODEL_URL es incorrecto o no accesible)
        print(f"Error de conexión con Hugging Face API: {e}")
        return jsonify({
            "error": f"Error de conexión: {e}. Asegúrate de que MODEL_URL es correcto y accesible.",
            "details": "El servidor de IA no pudo conectarse con la API de Hugging Face."
        }), 503 # Service Unavailable
    except requests.exceptions.Timeout as e:
        # Maneja errores de tiempo de espera
        print(f"Tiempo de espera agotado con Hugging Face API: {e}")
        return jsonify({
            "error": "Tiempo de espera agotado al conectar con la IA. Por favor, inténtalo de nuevo.",
            "details": f"La solicitud a la API de Hugging Face excedió el tiempo de espera: {e}"
        }), 504 # Gateway Timeout
    except requests.exceptions.RequestException as e:
        # Maneja cualquier otro error general de solicitudes
        print(f"Error general al conectar con Hugging Face API: {e}")
        return jsonify({
            "error": f"Error al conectar con la IA: {e}. Por favor, inténtalo de nuevo más tarde.",
            "details": f"Ocurrió un error inesperado al realizar la solicitud: {e}"
        }), 500
    except Exception as e:
        # Maneja cualquier otro error inesperado en tu código
        print(f"Error interno del servidor: {e}")
        return jsonify({
            "error": f"Ocurrió un error inesperado en el servidor: {e}. Por favor, contacta al soporte.",
            "details": f"Un error interno desconocido ocurrió: {e}"
        }), 500

# Asegúrate de que la aplicación se ejecuta cuando el script es el principal
# debug=True es útil para el desarrollo, pero deberías deshabilitarlo en producción.
# host='0.0.0.0' hace que el servidor escuche en todas las interfaces de red,
# lo cual es necesario para que sea accesible desde fuera de localhost en un entorno de despliegue.
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", 5000)) # Usa el puerto de la variable de entorno si existe, sino 5000
