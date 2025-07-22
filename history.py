session_history_store = {}

def save_history(session_id: str, question: str, answer: str, model_name: str, timestamp: str, pdf_names: str):
    entry = {
        "question": question,
        "answer": answer,
        "model": model_name,
        "timestamp": timestamp,
        "pdf_names": pdf_names,
    }
    session_history_store.setdefault(session_id, []).append(entry)

def get_history(session_id: str) -> list:
    return session_history_store.get(session_id, [])

def clear_history(session_id: str):
    session_history_store.pop(session_id, None) 