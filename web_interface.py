import os
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from rag_search import get_response


def generate(user_message):
    resp, score = get_response(user_message)
    if resp is None:
        return ("Я слышу тебя. Расскажи подробнее, что происходит? "
                "Я здесь, чтобы поддержать.")
    return resp


HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mental Health AI</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:linear-gradient(135deg,#667eea,#764ba2);min-height:100vh;display:flex;justify-content:center;align-items:center;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:20px}
  .container{background:rgba(255,255,255,0.95);border-radius:24px;box-shadow:0 20px 60px rgba(0,0,0,0.3);width:100%;max-width:600px;overflow:hidden;backdrop-filter:blur(10px)}
  .header{background:linear-gradient(135deg,#667eea,#764ba2);padding:24px 28px;color:#fff}
  .header h1{margin:0;font-size:22px;font-weight:600}
  .header p{margin:6px 0 0;font-size:13px;opacity:.85}
  .chat{height:400px;overflow-y:auto;padding:20px;background:#f8f9ff}
  .message{margin-bottom:16px;display:flex;flex-direction:column}
  .message.user{align-items:flex-end}
  .message.bot{align-items:flex-start}
  .bubble{max-width:80%;padding:12px 16px;border-radius:16px;font-size:14px;line-height:1.5;word-wrap:break-word}
  .user .bubble{background:#667eea;color:#fff;border-bottom-right-radius:4px}
  .bot .bubble{background:#fff;color:#333;border-bottom-left-radius:4px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
  .bot .bubble .typing{display:inline-block;animation:blink 1.4s infinite}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
  .input-area{display:flex;padding:16px;border-top:1px solid #e8ecf4;background:#fff}
  .input-area input{flex:1;padding:12px 16px;border:2px solid #e8ecf4;border-radius:12px;font-size:14px;outline:none;transition:border-color .2s}
  .input-area input:focus{border-color:#667eea}
  .input-area button{margin-left:10px;padding:12px 24px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border:none;border-radius:12px;font-size:14px;font-weight:600;cursor:pointer;transition:opacity .2s}
  .input-area button:hover{opacity:.9}
  .input-area button:disabled{opacity:.5;cursor:not-allowed}
  .disclaimer{text-align:center;padding:12px;font-size:11px;color:#999;border-top:1px solid #e8ecf4}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Mental Health AI</h1>
    <p>Терапевтический ИИ-помощник</p>
  </div>
  <div class="chat" id="chat"></div>
  <div class="input-area">
    <input type="text" id="input" placeholder="Напиши что тебя беспокоит..." autofocus>
    <button id="sendBtn">Отправить</button>
  </div>
  <div class="disclaimer">Не заменяет профессиональную помощь. При кризисе звони 8-800-2000-122</div>
</div>
<script>
  const chat=document.getElementById('chat'),input=document.getElementById('input'),sendBtn=document.getElementById('sendBtn');
  function addMsg(text,role){const d=document.createElement('div');d.className='message '+role;
    const b=document.createElement('div');b.className='bubble';b.textContent=text;
    d.appendChild(b);chat.appendChild(d);chat.scrollTop=chat.scrollHeight}
  function setLoading(l){sendBtn.disabled=l;input.disabled=l;input.focus()}
  async function send(){const msg=input.value.trim();if(!msg)return;input.value='';
    addMsg(msg,'user');setLoading(true);
    const d=document.createElement('div');d.className='message bot';
    const b=document.createElement('div');b.className='bubble';
    const s=document.createElement('span');s.className='typing';s.textContent='...';
    b.appendChild(s);d.appendChild(b);chat.appendChild(d);chat.scrollTop=chat.scrollHeight;
    try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
      const j=await r.json();b.textContent=j.response}catch(e){b.textContent='Ошибка соединения'}
    setLoading(false)}
  sendBtn.onclick=send;input.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey)send()};
  addMsg('Здравствуйте! Расскажите, что вас беспокоит. Я здесь, чтобы выслушать и поддержать.','bot');
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
            response = generate(body["message"])
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"response": response}, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    from rag_search import load_index
    load_index()
    port = 8765
    print(f"\n========================================")
    print(f" Mental Health AI запущен!")
    print(f"  http://localhost:{port}")
    print(f"  Режим: RAG поиск по датасету")
    print(f"========================================")
    print(f"\nНажми Ctrl+C для выхода\n")
    webbrowser.open(f"http://localhost:{port}")
    server = HTTPServer(("0.0.0.0", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nВеб-сервер остановлен.")
        server.shutdown()


if __name__ == "__main__":
    main()
