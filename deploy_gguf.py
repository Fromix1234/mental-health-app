import os
import sys
import json
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler

GGUF_MODEL = os.environ.get("GGUF_MODEL", "")
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))
PORT = int(os.environ.get("PORT", "8765"))

llm = None
tokenizer = None

def load_model(model_path):
    global llm
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Установи: pip install llama-cpp-python")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Модель не найдена: {model_path}")
        print("Сначала выполни: python finetune_qwen_lora.py")
        sys.exit(1)

    size_gb = os.path.getsize(model_path) / (1024**3)
    print(f"Загружаю GGUF модель: {model_path} ({size_gb:.2f} GB)")

    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    print("Модель загружена!")

SYSTEM_PROMPT = "Ты — опытный психолог с тёплым поддерживающим стилем. Отвечай на русском языке."

def generate_response(user_message):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.15,
        stop=["<|im_end|>"],
    )

    return response["choices"][0]["message"]["content"].strip()

HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mental Health AI — 9B</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{min-height:100vh;display:flex;justify-content:center;align-items:center;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',sans-serif;padding:20px;background:#f7f4ef;position:relative;overflow:hidden}
  body::before{content:'';position:fixed;top:-40%;left:-20%;width:80%;height:80%;background:radial-gradient(circle,rgba(124,154,124,0.15) 0%,transparent 70%);pointer-events:none}
  body::after{content:'';position:fixed;bottom:-30%;right:-20%;width:70%;height:70%;background:radial-gradient(circle,rgba(74,140,140,0.12) 0%,transparent 70%);pointer-events:none}
  .container{background:rgba(255,255,255,0.88);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border-radius:28px;box-shadow:0 8px 40px rgba(0,0,0,0.06),0 2px 8px rgba(0,0,0,0.04);width:100%;max-width:620px;overflow:hidden;border:1px solid rgba(255,255,255,0.6);position:relative}
  .header{padding:28px 28px 20px;text-align:center}
  .header-icon{width:52px;height:52px;background:linear-gradient(135deg,#7c9a7c,#4a8c8c);border-radius:16px;display:flex;align-items:center;justify-content:center;margin:0 auto 12px;box-shadow:0 4px 12px rgba(74,140,140,0.25)}
  .header-icon svg{width:28px;height:28px;fill:#fff}
  .header h1{margin:0;font-size:20px;font-weight:700;color:#2d3a2d;letter-spacing:-0.3px}
  .header p{margin:4px 0 0;font-size:13px;color:#8a9a8a;font-weight:450}
  .chat{height:420px;overflow-y:auto;padding:8px 20px 4px;background:rgba(247,244,239,0.5);scroll-behavior:smooth}
  .chat::-webkit-scrollbar{width:4px}
  .chat::-webkit-scrollbar-track{background:transparent}
  .chat::-webkit-scrollbar-thumb{background:#d4d0ca;border-radius:4px}
  .message{margin-bottom:12px;display:flex;flex-direction:column;opacity:0;animation:msgIn 0.3s ease forwards}
  @keyframes msgIn{to{opacity:1}}
  .message.user{align-items:flex-end}
  .message.bot{align-items:flex-start}
  .bubble{max-width:78%;padding:14px 18px;border-radius:20px;font-size:14.5px;line-height:1.6;word-wrap:break-word;transition:transform 0.15s}
  .user .bubble{background:linear-gradient(135deg,#7c9a7c,#4a8c8c);color:#fff;border-bottom-right-radius:4px;box-shadow:0 2px 8px rgba(74,140,140,0.2)}
  .bot .bubble{background:#fff;color:#2d3a2d;border-bottom-left-radius:4px;box-shadow:0 2px 8px rgba(0,0,0,0.04),0 1px 2px rgba(0,0,0,0.02);border:1px solid rgba(0,0,0,0.04)}
  .bot .bubble .typing{display:inline-flex;gap:4px;align-items:center;padding:2px 0}
  .typing span{width:7px;height:7px;background:#c4cfc4;border-radius:50%;display:inline-block;animation:typing 1.4s infinite both}
  .typing span:nth-child(2){animation-delay:0.2s}
  .typing span:nth-child(3){animation-delay:0.4s}
  @keyframes typing{0%,60%,100%{transform:scale(0.6);opacity:0.4}30%{transform:scale(1);opacity:1}}
  .input-area{display:flex;gap:8px;padding:12px 16px 16px;background:#fff;border-top:1px solid rgba(0,0,0,0.04)}
  .input-area input{flex:1;padding:14px 18px;border:2px solid #e8e4de;border-radius:14px;font-size:14px;outline:none;transition:border-color .25s,box-shadow .25s;background:#faf8f5;color:#2d3a2d;font-family:inherit}
  .input-area input::placeholder{color:#b8b2aa}
  .input-area input:focus{border-color:#7c9a7c;box-shadow:0 0 0 3px rgba(124,154,124,0.12);background:#fff}
  .input-area button{width:48px;height:48px;border:none;border-radius:14px;background:linear-gradient(135deg,#7c9a7c,#4a8c8c);color:#fff;cursor:pointer;transition:transform .15s,box-shadow .2s;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 4px 12px rgba(74,140,140,0.25)}
  .input-area button:hover{transform:translateY(-1px);box-shadow:0 6px 16px rgba(74,140,140,0.3)}
  .input-area button:active{transform:translateY(0);box-shadow:0 2px 6px rgba(74,140,140,0.2)}
  .input-area button:disabled{opacity:0.4;cursor:not-allowed;transform:none}
  .input-area button svg{width:20px;height:20px;fill:#fff;margin-left:2px}
  .disclaimer{text-align:center;padding:10px 16px;font-size:11px;color:#b8b2aa;background:#faf8f5;border-top:1px solid rgba(0,0,0,0.03);line-height:1.4}
  .welcome-msg{margin:20px 0;text-align:center}
  .welcome-msg .avatar{width:48px;height:48px;background:linear-gradient(135deg,#7c9a7c,#4a8c8c);border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 12px;box-shadow:0 4px 12px rgba(74,140,140,0.2)}
  .welcome-msg .avatar svg{width:24px;height:24px;fill:#fff}
  .welcome-msg p{font-size:14px;color:#8a9a8a;max-width:300px;margin:0 auto;line-height:1.5}
  .message-time{font-size:10px;color:#b8b2aa;margin:4px 8px 0}
  .user .message-time{text-align:right}
  @media(max-width:480px){body{padding:12px;align-items:stretch}
  .container{border-radius:20px}
  .header{padding:20px 20px 16px}
  .header-icon{width:44px;height:44px}
  .header-icon svg{width:24px;height:24px}
  .chat{height:calc(100vh - 240px);padding:4px 14px 2px}
  .bubble{max-width:88%;padding:12px 16px;font-size:14px}
  .input-area{padding:10px 12px 14px;gap:6px}
  .input-area input{padding:12px 14px;font-size:14px}
  .input-area button{width:44px;height:44px}}
  @media(prefers-color-scheme:dark){body{background:#1c241c}
  body::before{background:radial-gradient(circle,rgba(124,154,124,0.08) 0%,transparent 70%)}
  body::after{background:radial-gradient(circle,rgba(74,140,140,0.06) 0%,transparent 70%)}
  .container{background:rgba(30,38,30,0.92);backdrop-filter:blur(20px);border-color:rgba(255,255,255,0.06)}
  .header h1{color:#d4dfd4}
  .header p{color:#7a8a7a}
  .chat{background:rgba(26,34,26,0.5)}
  .chat::-webkit-scrollbar-thumb{background:#3a4a3a}
  .user .bubble{background:linear-gradient(135deg,#5a7a5a,#3a6a6a)}
  .bot .bubble{background:#263026;color:#d4dfd4;border-color:rgba(255,255,255,0.06)}
  .input-area{background:#263026;border-top-color:rgba(255,255,255,0.04)}
  .input-area input{background:#1c241c;border-color:#3a4a3a;color:#d4dfd4}
  .input-area input::placeholder{color:#5a6a5a}
  .input-area input:focus{border-color:#5a7a7a;background:#1c241c}
  .disclaimer{background:#1c241c;color:#5a6a5a}}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="header-icon">
      <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
    </div>
    <h1>Mental Health AI</h1>
    <p>9B модель · Qwen 2.5 · Q4_K_M</p>
  </div>
  <div class="chat" id="chat">
    <div class="welcome-msg">
      <div class="avatar">
        <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
      </div>
      <p>Здравствуйте! Расскажите, что вас беспокоит. Я здесь, чтобы выслушать и поддержать.</p>
    </div>
  </div>
  <div class="input-area">
    <input type="text" id="input" placeholder="Напишите сообщение..." autofocus>
    <button id="sendBtn" aria-label="Отправить">
      <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
    </button>
  </div>
  <div class="disclaimer">Не заменяет профессиональную помощь. При кризисе звоните 8-800-2000-122</div>
</div>
<script>
  const chat=document.getElementById('chat'),input=document.getElementById('input'),sendBtn=document.getElementById('sendBtn');
  function addMsg(text,role,dontScroll){const d=document.createElement('div');d.className='message '+role;
    const b=document.createElement('div');b.className='bubble';b.textContent=text;
    d.appendChild(b);const t=document.createElement('div');t.className='message-time';
    const now=new Date();t.textContent=now.getHours().toString().padStart(2,'0')+':'+now.getMinutes().toString().padStart(2,'0');
    d.appendChild(t);chat.appendChild(d);if(!dontScroll)chat.scrollTop=chat.scrollHeight}
  function setLoading(l){sendBtn.disabled=l;input.disabled=l;input.focus()}
  function showTyping(){const d=document.createElement('div');d.className='message bot';d.id='typing-indicator';
    const b=document.createElement('div');b.className='bubble';
    const t=document.createElement('span');t.className='typing';
    t.innerHTML='<span></span><span></span><span></span>';
    b.appendChild(t);d.appendChild(b);chat.appendChild(d);chat.scrollTop=chat.scrollHeight}
  function hideTyping(){const el=document.getElementById('typing-indicator');if(el)el.remove()}
  function scrollToBottom(){chat.scrollTop=chat.scrollHeight}
  async function send(){const msg=input.value.trim();if(!msg||sendBtn.disabled)return;input.value='';
    addMsg(msg,'user');setLoading(true);showTyping();scrollToBottom();
    try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
      const j=await r.json();hideTyping();addMsg(j.response,'bot')}
    catch(e){hideTyping();addMsg('Ошибка соединения. Проверьте подключение.','bot')}
    setLoading(false);scrollToBottom()}
  sendBtn.onclick=send;input.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey)send()};
</script>
</body>
</html>"""

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/chat":
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            response = generate_response(body["message"])
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"response": response}, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def find_gguf():
    paths = [
        GGUF_MODEL,
        "models/unsloth.Q4_K_M.gguf",
        "models/mental_health_ai.gguf",
    ]
    for p in paths:
        if p and os.path.exists(p):
            return p
    gguf_dir = "models"
    if os.path.isdir(gguf_dir):
        for f in os.listdir(gguf_dir):
            if f.endswith(".gguf"):
                return os.path.join(gguf_dir, f)
    return None

def main():
    model_path = find_gguf()
    if not model_path:
        print("GGUF модель не найдена. Укажи путь через GGUF_MODEL=... или запусти сперва:")
        print("  python finetune_qwen_lora.py")
        sys.exit(1)

    load_model(model_path)

    print(f"\nСервер запущен на http://localhost:{PORT}")
    print("Нажми Ctrl+C для остановки\n")

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
        server.shutdown()

if __name__ == "__main__":
    main()
