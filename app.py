import datetime, subprocess
import gradio as gr
from chat_logic import generate_response
from utils import (
    read_document, transcribe_audio,
    is_allowed_file, ALLOWED_DOC_EXTENSIONS, ALLOWED_AUDIO_EXTENSIONS
)

USERS        = {"admin": {"pwd": "admin123", "role": "admin"}, "user": {"pwd": "user123",  "role": "user"}}
CHAT_LOG     = []      
ADMIN_PARAMS = {"model": "ydharm/Lama", "temperature": 0.7}

def record_log(user, prompt):
    CHAT_LOG.append((user, prompt, datetime.datetime.now().isoformat()))

def build_prompt(txt, doc, aud):
    parts = []
    if txt:
        parts.append(txt.strip())
    if doc and is_allowed_file(doc.name, ALLOWED_DOC_EXTENSIONS):
        parts.append(read_document(doc.name))
    if aud and is_allowed_file(aud, ALLOWED_AUDIO_EXTENSIONS):
        parts.append(transcribe_audio(aud))
    return "\n".join(parts).strip()

SESSION = {}
def set_session(u: str | None = None, r: str | None = None):
    if u is None:
        SESSION.clear()
    else:
        SESSION["username"] = u
        SESSION["role"]     = r

def login_action(u, p):
    info = USERS.get(u)
    if info and info["pwd"] == p:
        set_session(u, info["role"])
        who = f"Logged in as **{u}**"
        admin_vis = True if info["role"] == "admin" else False
        return (
            gr.update(visible=False),  
            gr.update(value=who),          
            gr.update(visible=True),       
            gr.update(visible=True),
            gr.update(visible=admin_vis),    
            ""
        )
    err = "<span style='color:#f87171'> Invalid username / password</span>"
    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), err

def signup_action(u, p):
    if u in USERS:
        return "<span style='color:#f87171'> Username exists!</span>"
    USERS[u] = {"pwd": p, "role": "user"}
    return "<span style='color:#4ade80'> Signed up! Please log in.</span>"

def logout_action():
    set_session()
    return (
        gr.update(visible=True),
        gr.update(value=""),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False)
    )

def chat_action(msg, hist, doc, aud):
    user = SESSION.get("username", "anon")
    prompt = build_prompt(msg, doc, aud)
    if not prompt:
        return "", hist + [[msg, "Please enter text or upload."]]

    raw_html = generate_response(prompt)
    record_log(user, prompt)
    reply = f"```html\n{raw_html}\n```"
    return "", hist + [[msg, reply]]

def save_admin_settings(model, temp):
    ADMIN_PARAMS["model"] = model
    ADMIN_PARAMS["temperature"] = temp
    return f"✔️ Saved (model = `{model}`, temperature = {temp})"

def trigger_quick_train():
    subprocess.Popen(["python", "quick_train.py"])
    return " Quick‑train started!"

def view_logs():
    if not CHAT_LOG:
        return "No logs yet."
    return "\n".join(f"{ts} ▸ {u}: {p[:60]}…" for u, p, ts in CHAT_LOG[-100:])

custom_css = """
#title        {text-align:center; font-size:24px; font-weight:700; margin-bottom:12px;}
.gradio-container {background:#1f2937 !important; color:white;}

#login-card   {margin:auto; max-width:340px; padding:24px; border:1px solid #374151;
               border-radius:12px; background:#111827;}
.login-field input {background:#374151 !important; color:white;}

#header-bar   {justify-content:space-between; align-items:center; margin-bottom:6px;}
"""

with gr.Blocks(css=custom_css, title="Rapid_AI_Learning") as app:
    gr.Markdown("<div id='title'> Rapid_AI_Learning")

    with gr.Row(visible=True, elem_id="login-card") as login_row:
        with gr.Column():
            uname = gr.Textbox(label="Username", elem_classes="login-field")
            pwd   = gr.Textbox(label="Password", type="password",
                               elem_classes="login-field")
            with gr.Row():
                login_btn  = gr.Button("Login",  variant="primary")
                signup_btn = gr.Button("Signup")
            login_info = gr.Markdown("")

    with gr.Row(visible=False, elem_id="header-bar") as header_bar:
        whoami      = gr.Markdown("")
        logout_btn  = gr.Button("Logout", variant="secondary", size="sm")

    with gr.Tabs(visible=False) as main_tabs:

        with gr.Tab("💬 Chat"):
            chatbox = gr.Chatbot(height=420, show_copy_button=True)
            with gr.Row():
                txt = gr.Textbox(show_label=False, placeholder="Type message…",
                                 scale=4)
                doc = gr.File(label="PDF/DOCX",
                              file_types=[".pdf", ".docx"], scale=1)
                aud = gr.Audio(sources=["microphone"], type="filepath",
                               label="🎤", scale=1)
            txt.submit(chat_action, [txt, chatbox, doc, aud],
                       [txt, chatbox])

        with gr.Tab("🛠 Admin") as admin_tab:
            gr.Markdown("#### Model settings")
            mdl  = gr.Textbox(value=ADMIN_PARAMS["model"], label="Model name")
            tmp  = gr.Slider(0.1, 1.0, value=ADMIN_PARAMS["temperature"],
                             label="Temperature")
            save_btn  = gr.Button("Save settings", variant="primary")
            train_btn = gr.Button("Quick‑train model")
            admin_out = gr.Markdown("")
            gr.Markdown("#### Last 100 prompts")
            logs_box  = gr.TextArea(view_logs(), lines=15, interactive=False)

    login_btn.click(login_action,  [uname, pwd],
                    [login_row, whoami, header_bar, main_tabs, admin_tab, login_info])
    signup_btn.click(signup_action, [uname, pwd], login_info)
    logout_btn.click(logout_action, None,
                     [login_row, whoami, header_bar, main_tabs, admin_tab])

    save_btn.click(save_admin_settings, [mdl, tmp], admin_out)
    train_btn.click(trigger_quick_train,          None, admin_out)
    main_tabs.children[1].select(lambda: gr.update(value=view_logs()),
                                 None, logs_box)

app.queue()
app.launch(inline=True, share=True)
