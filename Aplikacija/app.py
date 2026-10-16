from dotenv import load_dotenv

load_dotenv()

from db.connection import STORE_INDEX_CONFIG, get_conn_string
from db.runtime import open_persistent_resources, run_coroutine, start_background_loop
from realestate.service import RealEstateSearchService
from ui.handlers import ChatHandlers
from ui.layout import build_ui

try:
    service = RealEstateSearchService("gradovi.db", use_llm=True, llm_model="gpt-4o-mini")
    LLM_ENABLED = True
except Exception as e:
    print(f" LLM not available: {e}")
    service = RealEstateSearchService("gradovi.db", use_llm=False)
    LLM_ENABLED = False

loop = start_background_loop()

checkpointer, store = run_coroutine(open_persistent_resources(get_conn_string(), STORE_INDEX_CONFIG))

handlers = ChatHandlers(service, checkpointer, store, loop)

demo = build_ui(handlers)

if __name__ == "__main__":
    print("Starting EstAIte...")
    print(f"LLM Mode: {'Enabled' if LLM_ENABLED else 'Disabled'}")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)