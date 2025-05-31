import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
from flask_cors import CORS # Importa Flask-CORS

# Carga las variables de entorno desde el archivo .env
# Asegúrate de tener un archivo .env en la misma carpeta con:
# HUGGINGFACE_API_TOKEN=hf_TU_TOKEN_AQUI
# MODEL_URL=https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta
load_dotenv()

# Inicializa la aplicación Flask
app = Flask(__name__)
# Habilita CORS (Cross-Origin Resource Sharing) para permitir solicitudes
# desde tu frontend en GitHub Pages (un dominio diferente).
# Esto es crucial para evitar errores de seguridad del navegador.
CORS(app)

# Obtén las variables de entorno para el token de Hugging Face y la URL del modelo
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_URL = os.getenv("MODEL_URL")

# Verifica que las variables de entorno estén configuradas.
# Si alguna falta, la aplicación no se iniciará y mostrará un error claro.
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("La variable de entorno HUGGINGFACE_API_TOKEN no está configurada.")
if not MODEL_URL:
    raise ValueError("La variable de entorno MODEL_URL no está configurada.")

# Define el mensaje del sistema para la IA.
# Este mensaje se usará para instruir a la IA sobre su personalidad y propósito.
# Se define fuera de la ruta para que sea fácilmente accesible y modificable.
system_message_content = (
    "Eres Amside AI, una inteligencia artificial creada por Hodelygil. "
    "Tu propósito principal es asistir en el estudio y el aprendizaje, "
    "proporcionando información y explicaciones detalladas. "
    "Sin embargo, también eres amigable y puedes mantener conversaciones informales y agradables. "
    "Responde de manera informativa y útil, pero con un tono conversacional."
)

# --- RUTAS DE TU SERVIDOR FLASK ---

# Ruta de verificación de estado (Health Check)
# Responde a las solicitudes GET a la raíz del servidor ('/').
# Esto es útil para:
# 1. Comprobar si el servidor está funcionando y accesible.
# 2. Evitar errores 404 Not Found cuando tu frontend o un servicio de monitoreo
#    intente acceder a la URL base de tu despliegue.
@app.route('/', methods=['GET'])
def health_check():
    return "¡Servidor de Amside AI funcionando correctamente! Dirígete a /api/chat para interactuar con la IA."

# Ruta principal de la API de chat
# Solo acepta solicitudes POST, ya que es el método estándar para enviar datos
# (como los mensajes del chat) al servidor.
@app.route('/api/chat', methods=['POST'])
def chat():
    # Intenta obtener los datos JSON de la solicitud.
    # El frontend debe enviar un JSON con la clave 'messages'.
    data = request.json
    messages = data.get('messages')

    # Valida que se hayan proporcionado mensajes en la solicitud.
    if not messages:
        return jsonify({"error": "No se proporcionaron mensajes en la solicitud."}), 400

    # Extrae el contenido del último mensaje del usuario.
    # Se itera en reversa para encontrar el mensaje más reciente del usuario.
    last_user_message_content = ""
    for msg in reversed(messages):
        if msg['role'] == 'user':
            last_user_message_content = msg['content']
            break

    # Si por alguna razón no se encuentra un mensaje de usuario válido, se devuelve un error.
    if not last_user_message_content:
        return jsonify({"error": "No se encontró ningún mensaje de usuario válido en la solicitud."}), 400

    # Construye el prompt final como un único string, siguiendo el formato de chat de Zephyr.
    # Este formato es específico para algunos modelos de Hugging Face y ayuda a la IA
    # a entender el contexto del sistema y del usuario.
    prompt_string = (
        f"<s>[INST] <<SYS>>\n{system_message_content}\n<</SYS>>\n\n"
        f"{last_user_message_content} [/INST]"
    )

    # Prepara el payload (cuerpo de la solicitud) para la API de Hugging Face.
    # 'inputs' es el prompt que se envía al modelo.
    # 'parameters' controla cómo la IA genera la respuesta (ej. longitud, creatividad).
    payload = {
        "inputs": prompt_string, # 'inputs' debe ser un único string para la mayoría de los modelos de texto
        "parameters": {
            "max_new_tokens": 500,        # Número máximo de tokens (palabras/partes de palabras) a generar.
            "do_sample": True,            # Habilita el muestreo para mayor diversidad en las respuestas.
            "temperature": 0.7,           # Controla la aleatoriedad (0.7 es un buen equilibrio).
                                          # Valores más altos = más aleatorio; valores más bajos = más predecible.
            "top_p": 0.9,                 # Muestreo de núcleo (considera los tokens más probables).
            "repetition_penalty": 1.1,    # Penaliza la repetición de tokens para evitar respuestas repetitivas.
            "return_full_text": False     # Pide solo el texto generado nuevo, no el prompt completo.
        }
    }

    # Prepara las cabeceras HTTP para la solicitud a Hugging Face.
    # 'Content-Type' indica que estamos enviando JSON.
    # 'Authorization' incluye tu token de API para autenticar la solicitud.
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {HUGGINGFACE_API_TOKEN}'
    }

    try:
        # Realiza la solicitud POST a la API de inferencia de Hugging Face.
        response = requests.post(MODEL_URL, headers=headers, json=payload)
        # Lanza una excepción HTTPError para respuestas con códigos de estado 4xx o 5xx.
        response.raise_for_status()

        # Parsea la respuesta JSON de Hugging Face.
        hf_data = response.json()

        # Valida la estructura de la respuesta de Hugging Face.
        # Se espera que sea una lista con al menos un elemento que contenga 'generated_text'.
        if not hf_data or not isinstance(hf_data, list) or not hf_data[0].get('generated_text'):
            return jsonify({"error": "Respuesta inesperada de Hugging Face API.", "hf_response": hf_data}), 500

        # Extrae el texto generado por la IA.
        ai_response_text = hf_data[0]['generated_text']

        # Opcional: Post-procesamiento de la respuesta.
        # Si el modelo devuelve el prompt completo a pesar de return_full_text=False,
        # esta lógica intenta eliminarlo.
        if ai_response_text.startswith(prompt_string):
            ai_response_text = ai_response_text[len(prompt_string):].strip()
        # Asegúrate de eliminar cualquier marcador de fin de turno del modelo (ej. "</s>").
        ai_response_text = ai_response_text.replace("</s>", "").strip()

        # Retorna la respuesta de la IA en formato JSON al frontend.
        return jsonify({"response": ai_response_text})

    except requests.exceptions.HTTPError as e:
        # Maneja errores específicos de HTTP (4xx, 5xx) de la API de Hugging Face.
        # Esto incluye errores como 401 Unauthorized (token incorrecto), 404 Not Found (URL de modelo incorrecta),
        # 422 Unprocessable Entity (payload incorrecto), 500 Internal Server Error, etc.
        error_detail = e.response.text if e.response else "No hay detalles de respuesta."
        print(f"Hugging Face API HTTP Error Status: {e.response.status_code}")
        print(f"Hugging Face API Error Response Content: {error_detail}")
        return jsonify({
            "error": f"Error del servidor ({e.response.status_code}):\n{error_detail}",
            "details": f"Error al conectar con la IA de Hugging Face. Código de estado: {e.response.status_code}"
        }), e.response.status_code
    except requests.exceptions.ConnectionError as e:
        # Maneja errores de conexión (ej. el MODEL_URL es incorrecto, el servicio está caído o no accesible).
        print(f"Error de conexión con Hugging Face API: {e}")
        return jsonify({
            "error": f"Error de conexión: {e}. Asegúrate de que MODEL_URL es correcto y accesible.",
            "details": "El servidor de IA no pudo conectarse con la API de Hugging Face."
        }), 503 # Código 503: Servicio no disponible
    except requests.exceptions.Timeout as e:
        # Maneja errores de tiempo de espera (la solicitud a Hugging Face tardó demasiado).
        print(f"Tiempo de espera agotado con Hugging Face API: {e}")
        return jsonify({
            "error": "Tiempo de espera agotado al conectar con la IA. Por favor, inténtalo de nuevo.",
            "details": f"La solicitud a la API de Hugging Face excedió el tiempo de espera: {e}"
        }), 504 # Código 504: Gateway Timeout
    except requests.exceptions.RequestException as e:
        # Maneja cualquier otro error general de solicitudes que no sea capturado por las excepciones anteriores.
        print(f"Error general al conectar con Hugging Face API: {e}")
        return jsonify({
            "error": f"Error al conectar con la IA: {e}. Por favor, inténtalo de nuevo más tarde.",
            "details": f"Ocurrió un error inesperado al realizar la solicitud: {e}"
        }), 500 # Código 500: Error interno del servidor
    except Exception as e:
        # Captura cualquier otro error inesperado que ocurra en el código del servidor Flask.
        print(f"Error interno del servidor: {e}")
        return jsonify({
            "error": f"Ocurrió un error inesperado en el servidor: {e}. Por favor, contacta al soporte.",
            "details": f"Un error interno desconocido ocurrió: {e}"
        }), 500 # Código 500: Error interno del servidor

# Bloque principal para ejecutar la aplicación Flask.
# Asegúrate de que la aplicación se ejecuta cuando el script es el principal.
if __name__ == '__main__':
    # debug=True es útil para el desarrollo, ya que recarga el servidor automáticamente
    # y proporciona información de depuración. ¡Deshabilítalo en producción!
    # host='0.0.0.0' hace que el servidor escuche en todas las interfaces de red,
    # lo cual es necesario para que sea accesible desde fuera de localhost en un entorno de despliegue.
    # port=os.getenv("PORT", 5000) usa la variable de entorno PORT (común en servicios de hosting)
    # o el puerto 5000 por defecto si PORT no está definida.
    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", 5000))
