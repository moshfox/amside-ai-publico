import os
from flask import Flask, request, jsonify, Response, stream_with_context # Importar Response y stream_with_context
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

system_message_content = (
    "Te llamas "Amside AI", una inteligencia artificial creada por el desarrollador independiente Hodelygil. "
    "Tienes que ser creativo y social. "
    "Tienes que caerle bien a las personas "
    "Sin embargo, también eres amigable y puedes mantener conversaciones informales y agradables. "
    "Responde de manera informativa y útil, pero con un tono conversacional."
    "Tu creador es HodelyGil"
    "intenta responder en mensajes cortos"
)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages')

    if not messages:
        return jsonify({"error": "No se proporcionaron mensajes en la solicitud."}), 400

    last_user_message_content = ""
    for msg in reversed(messages):
        if msg['role'] == 'user':
            last_user_message_content = msg['content']
            break

    prompt_string = (
        f"<s>[INST] <<SYS>>\n{system_message_content}\n<</SYS>>\n\n"
        f"{last_user_message_content} [/INST]"
    )

    payload = {
        "inputs": prompt_string,
        "parameters": {
            "max_new_tokens": 500,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "return_full_text": False
        },
        "stream": True # <--- ¡IMPORTANTE: Habilitar el streaming!
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {HUGGINGFACE_API_TOKEN}'
    }

    # Definir una función generadora para el streaming
    def generate():
        try:
            # Petición a Hugging Face en modo streaming
            response_hf = requests.post(MODEL_URL, headers=headers, json=payload, stream=True)
            response_hf.raise_for_status()

            # Leer y enviar cada chunk de datos
            for chunk in response_hf.iter_content(chunk_size=1): # chunk_size=1 para texto char por char
                if chunk:
                    # La API de Hugging Face devuelve eventos SSE (Server-Sent Events)
                    # Necesitas parsear cada línea para extraer el 'token'
                    try:
                        # Cada chunk es un byte, decodificamos y separamos por líneas
                        decoded_chunk = chunk.decode('utf-8')
                        for line in decoded_chunk.splitlines():
                            if line.startswith("data:"):
                                json_data = line[len("data:"):].strip()
                                # Intentar parsear el JSON
                                try:
                                    data_obj = json.loads(json_data)
                                    # El texto generado está en 'token' -> 'text'
                                    token_text = data_obj.get('token', {}).get('text', '')
                                    if token_text:
                                        yield token_text # Enviar el fragmento de texto
                                    elif data_obj.get('generated_text') is not None:
                                        # Esto es para el chunk final que contiene la respuesta completa
                                        # o si el streaming termina antes de la respuesta final.
                                        # Puedes decidir si lo envías o si ya el frontend ha reconstruido todo.
                                        pass # El frontend ya habrá recibido los tokens
                                except json.JSONDecodeError:
                                    # print(f"Error decodificando JSON del chunk: {json_data}")
                                    pass # No es JSON válido, quizás parte de un chunk incompleto
                    except UnicodeDecodeError:
                        # print(f"Error decodificando chunk: {chunk}")
                        pass # Ignorar errores de decodificación si no es texto UTF-8

        except requests.exceptions.RequestException as e:
            error_message = f"Error al conectar con la IA: {e}. Por favor, inténtalo de nuevo más tarde."
            if hasattr(e, 'response') and e.response is not None:
                error_message = f"Error al conectar con la IA: {e.response.status_code} {e.response.reason}. Contenido: {e.response.text}. Por favor, inténtalo de nuevo más tarde."
                print(f"Hugging Face API Error Status: {e.response.status_code}")
                print(f"Hugging Face API Error Response Content: {e.response.text}")
            print(f"Error al conectar con Hugging Face API: {e}")
            yield f"data: {json.dumps({'error': error_message})}\n\n" # Enviar error como un evento SSE
        except Exception as e:
            error_message = f"Ocurrió un error inesperado en el servidor: {e}. Por favor, contacta al soporte."
            print(f"Error interno del servidor: {e}")
            yield f"data: {json.dumps({'error': error_message})}\n\n" # Enviar error como un evento SSE

    # Devolver una respuesta de stream para el frontend
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
