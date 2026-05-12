import torch
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)

#서버 시작하면 모델을 한번만 로드 -> 여러번 작동 시 메모리 누수 위험
MODEL_NAME = "facebook/nllb-200-distilled-600M" #번역을 위한 모델 선언
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"모델 로딩 중... (Device: {device})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
print("모델 로딩 완료!")

def translate_logic(text):
    """
    미리 로드된 모델을 사용하여 번역을 수행합니다.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # NLLB 언어 코드 설정 (영어 -> 한국어)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("kor_Hang")

        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256
        )

        result = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return result
    except Exception as e:
        return f"번역 오류: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/speech')
def speech_ui():
    return render_template('speech.html')

@app.route('/camera')
def camera_ui():
    return render_template('camera.html')

@app.route('/api/translate', methods=['POST'])
def handle_translation():
    data = request.get_json()
    english_text = data.get('text', '').strip()

    if not english_text:
        return jsonify({"status": "error", "message": "인식된 텍스트가 없습니다."})

    try:
        # 번역 실행
        translated_text = translate_logic(english_text)
        print(f"인식: {english_text} -> 번역: {translated_text}")

        return jsonify({
            "status": "ok",
            "original": english_text,
            "translated": translated_text
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # debug=True 모드에서는 모델이 두 번 로드될 수 있으므로 주의하세요.
    app.run(host='0.0.0.0', port=5000)
    # app.run(host='0.0.0.0', port=5000, ssl_context="adhoc") # 로컬 https 로 열려면 사용 (안전하지 않음으로 나옴, 경고 무시하고 접속해야 함)