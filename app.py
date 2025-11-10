# app.py
"""
Hugging Face Space frontend for Brahmaanu LLM

This script recreates the original Gradio UI from the main repository
but delegates all heavy lifting to a remote API.  The API should be
deployed separately (e.g. via the modified Dockerfile) and exposes
``/chat`` and ``/sample_questions`` endpoints as defined in
``app/api.py``.

To run this Space locally, set the ``BACKEND_URL`` environment variable to
point at your API server.  When deployed as a Hugging Face Space,
``BACKEND_URL`` can be configured via the Space settings or secrets.

The design, theme, CSS and layout mirror the original ``main_gradio.py``
file in the GitHub repo to maintain a consistent user experience.
"""

import os
from typing import List, Tuple, Dict, Any
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError
import gradio as gr


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Base URL for the Brahmaanu API.  Use an environment variable so it can
# be configured without editing the code.  Defaults to localhost for
# convenience during development.  Do not include a trailing slash.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7861").rstrip("/")
print(f"[HF] BACKEND_URL resolved to: {BACKEND_URL}")

# Model modes supported by the backend.  Keep these in sync with the
# server‑side definition in app/main_gradio.py.
MODES = ["SFT_RAG", "SFT", "BASE_RAG", "BASE"]

# Health check
def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=10)
        r.raise_for_status()
        if r.ok and r.json().get("status") == "ok":
            return "backend=READY"
    except Exception:
        pass
    return "backend=DOWN"
    

# -----------------------------------------------------------------------------
# Sample questions helper
# -----------------------------------------------------------------------------
def fetch_sample_questions() -> List[str]:
    """Attempt to fetch sample questions from the API, falling back to static."""
    try:
        resp = requests.get(f"{BACKEND_URL}/sample_questions", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # Fallback to static list copied from the original UI
    return [
        "Where is the observatory located?",
        "Describe the governance settings for observing cycle length, cycle start date, cycle end date, and director’s discretionary time per cycle?",
        "Outline the key policies for observing modes, observing queue categories, and remote observation support?",
        "Summarize the discovery details for Dhruva Exoplanet and Aditi Pulsar?",
        "Give the deepimager exposure time range?",
        "Summarize the key calibration settings for flat field frames per filter per night, flat field illumination source, viscam twilight flat window, and flat field exposure level?",
    ]


# -----------------------------------------------------------------------------
# Chat proxy
# -----------------------------------------------------------------------------
def call_api(
    user_msg: str,
    chat_history: List[Tuple[str, str]],
    mode: str,
    use_history: bool,
    state: Dict[str, Any],
) -> Tuple[str, List[Tuple[str, str]], Dict[str, Any], str]:
    """
    Proxy the chat request to the remote API.

    Returns a tuple of (empty_prompt, chat_history, state, status) to match
    the signature expected by Gradio event handlers.
    """
    payload = {
        "user_msg": user_msg,
        "chat_history": chat_history or [],
        "mode": mode,
        "use_history": use_history,
        "state": state or {},
    }
    url = f"{BACKEND_URL}/chat"
    print(f"[HF] POST {url}")
    print(f"[HF] payload keys = {list(payload.keys())}")
    try:
        resp = requests.post(url, json=payload, timeout=60, allow_redirects=False)
        print(f"[HF] status = {resp.status_code}")
        if resp.is_redirect:
            print(f"[HF] redirect → {resp.headers.get('location')}")
        resp.raise_for_status()
        data = resp.json()
        print(f"[HF] response keys = {list(data.keys())}")
        # Do not clear the textbox after sending.  Returning the original
        # user_msg keeps the prompt in place so users can re‑submit or edit it,
        # and prevents a subsequent empty prompt from being sent:contentReference[oaicite:2]{index=2}.
        return (
         user_msg,
         data.get("chat_history", chat_history),
         data.get("state", state),
         data.get("status", ""),
        )
        
    except ConnectionError as e:
        # typically backend not up / refused
        err_msg = "Backend not reachable (connection refused). Please retry when the backend server is running..."
        new_hist = list(chat_history)
        new_hist.append((user_msg, err_msg))
        return user_msg, new_hist, state or {}, err_msg

    except Timeout:
        err_msg = "Backend took too long. Please try again later."
        new_hist = list(chat_history)
        new_hist.append((user_msg, err_msg))
        return user_msg, new_hist, state or {}, err_msg

    except HTTPError as e:
        err_msg = f"Backend returned HTTP Error : {e.response.status_code}."
        new_hist = list(chat_history)
        new_hist.append((user_msg, err_msg))
        return user_msg, new_hist, state or {}, err_msg

    except Exception:
        err_msg = "Unknown client error."
        new_hist = list(chat_history)
        new_hist.append((user_msg, err_msg))
        return user_msg, new_hist, state or {}, err_msg


# -----------------------------------------------------------------------------
# UI definition
# -----------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    """Construct the Gradio interface."""
    sample_questions = fetch_sample_questions()
    # Sci‑fi-ish background; allow override via BRAHMAANU_BG env
    bg_img = os.getenv(
        "BRAHMAANU_BG",
        "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?auto=format&fit=crop&w=1800&q=50",
    )
    with gr.Blocks(
        title="Brahmaanu LLM · Chat",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=f"""
        body {{
            font-family: "Segoe UI", "Roboto", "system-ui", -apple-system, BlinkMacSystemFont, sans-serif;
            background: #020617;
        }}
        body::before {{
            content: "";
            position: fixed;
            inset: 0;
            background:
                linear-gradient(135deg, rgba(2,6,23,0.35), rgba(2,6,23,0.9)),
                url('{bg_img}') center/cover no-repeat fixed;
            z-index: -2;
            filter: saturate(1);
        }}
        .gradio-container {{
            max-width: 1180px !important;
            margin: 0 auto !important;
        }}
        #top-card {{
            background: radial-gradient(circle at 10% 20%, #0f2f6b 0%, #020617 60%);
            padding: 14px 16px 10px 16px;
            border-radius: 14px;
            color: #ffffff;
            box-shadow: 0 6px 18px rgba(0,0,0,0.35);
            margin-bottom: 14px;
            border: 1px solid rgba(139,162,191,0.4);
        }}
        #top-title {{
            font-weight: 650;
            font-size: 1.04rem;
            letter-spacing: 0.04em;
            color: #ffffff !important;
        }}
        #top-title p {{
            color: #ffffff !important;
        }}
        #top-title p strong {{
            color: #ffffff !important;
        }}
        #controls-card {{
            background: rgba(241,245,249,0.9);
            border: 1px solid #d0d7e2;
            border-radius: 12px;
            padding: 10px 10px 6px 10px;
            margin-bottom: 10px;
            color: #0f172a;
        }}
        #use-hist-box {{
            background: rgba(226,232,240,0.95);
            border: 1px solid rgba(148,163,184,0.65);
            border-radius: 8px;
            padding: 4px 8px 2px 8px;
        }}
        #input-card {{
            background: rgba(243,244,246,0.9);
            border: 1px solid #d1d5db;
            border-radius: 12px;
            padding: 10px 10px 12px 10px;
            margin-bottom: 10px;
        }}
        #chat-card {{
            background: rgba(248,250,252,0.98);
            border: 1px solid #cbd5f5;
            border-radius: 14px;
            padding: 6px;
            box-shadow: 0 4px 12px rgba(15,23,42,0.08);
        }}
        .chatbot {{
            border: none !important;
            background: #ffffff !important;
        }}
        input[type="checkbox"] {{
            accent-color: #0f5fff;
            border: 1px solid #000000 !important;
            box-shadow: 0 0 1px #000000 !important;
        }}
        """,
    ) as demo:
        with gr.Column():
            # top header
            with gr.Row(elem_id="top-card"):
                gr.Markdown(
                    "Brahmaanu LLM · Mistral-7B SFT  RAG · **by Srivatsava Kasibhatla**",
                    elem_id="top-title",
                )

            # mode  history  status
            with gr.Row(elem_id="controls-card"):
                mode_dd = gr.Dropdown(choices=MODES, value="SFT_RAG", label="Mode")
                with gr.Column(elem_id="use-hist-box", scale=2):
                    use_hist = gr.Checkbox(
                        value=False,
                        label="Use conversation history for this request",
                    )
                # status label – intial state
                status_lbl = gr.Markdown("mode=SFT_RAG · memory=0/10 · checking backend...")


            # state holds session id and memory; created here and bound later
            state = gr.State({"session_id": "", "memory": []})

            # input section
            with gr.Column(elem_id="input-card"):
                sample_dd = gr.Dropdown(
                    choices=sample_questions,
                    value=None,
                    label="Sample questions",
                    interactive=True,
                )
                msg = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Ask a handbook question...",
                )
                send = gr.Button("Send", variant="primary")
                
                def _pick_sample(q: str) -> Tuple[str, None]:
                    """
                    Populate the prompt with the selected sample question and reset the
                    dropdown.  Resetting the dropdown allows the same question to be
                    selected again without refreshing:contentReference[oaicite:3]{index=3}.
                    """
                    return (q or "", None)
                # Update both the textbox and the dropdown value when a sample is chosen.
                sample_dd.change(_pick_sample, inputs=sample_dd, outputs=[msg, sample_dd])


            # chat history at bottom
            with gr.Row(elem_id="chat-card"):
                chat = gr.Chatbot(
                    height=520,
                    label="Chat",
                    elem_classes=["chatbot"],
                )

            # define submit callback that proxies to API
            def _submit(user_msg, chat_hist, mode, use_h, st):
                return call_api(user_msg, chat_hist, mode, use_h, st)

            # wire send button and Enter key to callback
            send.click(
                _submit,
                [msg, chat, mode_dd, use_hist, state],
                [msg, chat, state, status_lbl],
            )
            msg.submit(
                _submit,
                [msg, chat, mode_dd, use_hist, state],
                [msg, chat, state, status_lbl],
            )
            # after layout is built, ping backend once
            demo.load(
                fn=lambda: check_backend(),
                inputs=None,
                outputs=status_lbl,
            )

        return demo


if __name__ == "__main__":
    interface = build_ui()
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    share = os.getenv("GRADIO_PUBLIC_SHARE", "True").lower() == "true"
    interface.launch(server_name="0.0.0.0", server_port=port, share=share)