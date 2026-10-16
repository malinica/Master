import gradio as gr

from ui.theme import CSS

EXAMPLES = [
    "Cheap and sunny towns in Oklahoma",
    "Warm affordable properties in California with good schools",
    "Luxury homes in Florida without snow",
    "Find me budget-friendly towns in Texas with elementary schools",
    "Expensive towns in Washington with far hospitals",
    "Moderate price homes in Colorado with good education",
    "Cheapest towns",
    "Hottest towns",
    "Towns with price about 300k in Nevada",
    "Sunny places in California",
    "Towns with temperature beetwen 10 and 15 degrees",
    "Find me places with hospitals within 5km in Minnesota",
    "Find me places with schools within 20km in Nebraska"
]


def build_ui(handlers) -> gr.Blocks:
    with gr.Blocks(theme="base", css=CSS) as demo:

        with gr.Row(elem_id="main-container"):

            with gr.Column(elem_id="sidebar"):
                with gr.Column():
                    gr.HTML("""
                        <h1 id="title">
                            <span style="color:#3468C0;">Est</span><span style="color:#86A7FC;">AI</span><span style="color:#3468C0;">te</span>
                        </h1>
                    """)
                with gr.Column(elem_id="control-panel"):
                    user_id_state = gr.State(None)
                    thread_id_state = gr.State(None)
                    status_text = gr.Markdown("<span style='color: var(--muted);'>Enter both username and password to start</span>")

                    username_box = gr.Textbox(
                        placeholder="Username",
                        show_label=False,
                    )
                    password_box = gr.Textbox(
                        placeholder="Password",
                        show_label=False,
                        type="password",
                    )

                    with gr.Row():
                        login_btn = gr.Button("Login", elem_classes="control-btn", scale=1, size="sm")
                        logout_btn = gr.Button(
                            "Logout",
                            elem_classes="control-btn",
                            elem_id="logout-btn",
                            scale=1,
                            size="sm"
                        )

                    session_dropdown = gr.Dropdown(
                        label="Load Session",
                        choices=[],
                        value=None,
                        allow_custom_value=False
                    )

                    gr.Markdown(
                        "<span style='color: var(--muted); font-size: 0.85em;'>"
                        "New chat settings (locked in once a chat starts):</span>"
                    )
                    agent_variant_radio = gr.Radio(
                        ["baseline", "memory"],
                        label="Agent variant",
                        value="baseline",
                    )
                    memory_types_checkboxes = gr.CheckboxGroup(
                        ["semantic", "episodic", "procedural"],
                        label="Memory types (pick any combination)",
                        value=["semantic", "episodic", "procedural"],
                        visible=False,
                    )
                    extraction_mode_radio = gr.Radio(
                        ["hot_path", "background", "both"],
                        label="Memory extraction mode",
                        value="hot_path",
                        visible=False,
                    )
                    mode_lock_note = gr.Markdown(
                        "<span style='color: var(--muted); font-size: 0.85em;'>"
                        "hot_path/background only affects semantic and episodic memory — "
                        "procedural memory has no extraction mode, it just layers in the "
                        "latest optimized prompt if one exists.</span>",
                        visible=False,
                    )

                    def _update_memory_controls(agent_variant):
                        is_memory = agent_variant == "memory"
                        return (
                            gr.update(visible=is_memory),
                            gr.update(visible=is_memory),
                            gr.update(visible=is_memory),
                        )

                    agent_variant_radio.change(
                        _update_memory_controls,
                        [agent_variant_radio],
                        [memory_types_checkboxes, extraction_mode_radio, mode_lock_note],
                    )

                    new_chat_btn = gr.Button(
                        "New chat",
                        elem_classes="control-btn",
                        elem_id="new-chat-btn",
                        size="sm"
                    )
                    delete_btn = gr.Button(
                        "Delete session",
                        elem_classes="control-btn",
                        elem_id="delete-btn",
                        variant="stop",
                        size="sm"
                    )
                    delete_all_btn = gr.Button(
                        "Delete all",
                        elem_classes="control-btn",
                        elem_id="delete-all-btn",
                        variant="stop",
                        size="sm"
                    )

            with gr.Column(elem_id="chat-section"):
                with gr.Column(elem_id="chat-content"):
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        height=600,
                        show_label=False,
                        container=False
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask me about real estate...",
                            show_label=False,
                            lines=2,
                            scale=8,
                            interactive=False,
                        )
                        send = gr.Button(
                            "Send",
                            variant="primary",
                            elem_id="send-btn",
                            scale=1,
                            min_width=100,
                            interactive=False,
                        )

                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=msg,
                        label="Example queries"
                    )

        login_btn.click(
            handlers.login,
            [username_box, password_box],
            [user_id_state, status_text, session_dropdown, msg, send],
        )
        username_box.submit(
            handlers.login,
            [username_box, password_box],
            [user_id_state, status_text, session_dropdown, msg, send],
        )
        password_box.submit(
            handlers.login,
            [username_box, password_box],
            [user_id_state, status_text, session_dropdown, msg, send],
        )

        logout_btn.click(
            handlers.logout,
            [user_id_state],
            [user_id_state, thread_id_state, chatbot, session_dropdown, status_text, username_box, password_box, msg, send],
        )

        send.click(
            handlers.respond,
            [msg, chatbot, user_id_state, thread_id_state, agent_variant_radio, extraction_mode_radio, memory_types_checkboxes],
            [msg, chatbot, thread_id_state]
        ).then(
            handlers.list_threads,
            [user_id_state],
            session_dropdown
        )

        msg.submit(
            handlers.respond,
            [msg, chatbot, user_id_state, thread_id_state, agent_variant_radio, extraction_mode_radio, memory_types_checkboxes],
            [msg, chatbot, thread_id_state]
        ).then(
            handlers.list_threads,
            [user_id_state],
            session_dropdown
        )

        new_chat_btn.click(
            handlers.new_chat,
            [user_id_state],
            [chatbot, thread_id_state, session_dropdown]
        ).then(
            lambda: "New chat session started",
            [],
            status_text
        )

        session_dropdown.change(
            handlers.load_thread,
            [session_dropdown, user_id_state],
            [chatbot, thread_id_state]
        )

        delete_btn.click(
            handlers.delete_thread,
            [thread_id_state, user_id_state],
            [chatbot, thread_id_state, session_dropdown]
        ).then(
            lambda: "Session deleted",
            [],
            status_text
        )

        delete_all_btn.click(
            handlers.delete_all_threads,
            [user_id_state],
            [chatbot, thread_id_state, session_dropdown, status_text]
        )

    return demo
