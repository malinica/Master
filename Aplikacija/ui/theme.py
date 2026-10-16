CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
:root {
  --bg: #ffffff;
  --card: #ffffff;
  --muted: #6b6b6b;
  --accent: #86A7FC;
  --accent-2: #C2D3FD;
  --accent-3: #FFDD95;
  --accent-4: #3468C0;
  --accent-5: #FF9843;
}

body {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
  background-color: var(--bg) !important;
  color: var(--muted);
  margin: 0;
  padding: 0;
}

#main-container {
    display: flex;
    width: 100%;
    min-height: 100vh;
    gap: 0;
}

#sidebar {
    width: 370px;
    min-width: 370px;
    max-width: 370px;
    background-color: var(--card) !important;
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
}

#chat-section {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 40px 20px;
    overflow-y: auto;
}

#chat-content {
    width: 100%;
    max-width: 880px;
    margin: 0 auto;
}

#control-panel {
    padding: 18px;
    background-color: var(--bg) !important;
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 12px;
}

.control-btn {
    width: 100% !important;
    margin-top: 8px;
    margin-bottom: 8px;
    border-radius: 12px !important;
    padding: 10px !important;
    font-weight: 600;
}

#new-chat-btn {
    background: linear-gradient(0deg, var(--accent), var(--accent)) !important;
    color: #fff !important;
    border: none !important;
}

#logout-btn {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 12px;
}

#delete-btn {
    background: var(--accent-5) !important;
    color: var(--muted) !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 12px;
}

#delete-all-btn {
    background: var(--accent-3) !important;
    color: var(--muted) !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 12px;
}

#send-btn {
    background: var(--accent-4);
    border: none !important;
    color: #fff !important;
    height: 85px !important;
    border-radius: 12px !important;
    min-width: 100px !important;
}

#title {
    text-align: left;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    margin-top: -20px;
    margin-bottom: -20px;
}

.gradio-container {
    background: transparent;
    max-width: none !important;
}

#chatbot {
    background: transparent;
    border-radius: 12px;
    padding: 14px;
    margin-top: 50px;
}

.gradio-chat-message.user {
    background: var(--accent-2);
    color: var(--muted) !important;
    border-radius: 12px 12px 4px 12px;
    padding: 10px;
}

.gradio-chat-message.bot {
    background: var(--muted) !important;
    color: var(--muted) !important;
    border-radius: 12px 12px 12px 4px;
    padding: 10px;
}

textarea, input[type=text] {
    border-radius: 12px !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 12px;
    padding: 12px !important;
}

.gr-button { box-shadow: none !important; }

.gr-examples { background: transparent; }
.gradio-examples .gradio-card {
    border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.04);
}

@media (max-width: 800px) {
    #main-container {
        flex-direction: column;
    }
    #sidebar {
        width: 100%;
        border-right: none;
        border-bottom: 1px solid rgba(0,0,0,0.06);
    }
    #chat-section {
        padding: 20px 10px;
    }
}
"""
