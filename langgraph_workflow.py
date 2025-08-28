from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from llm_utils import get_chain
from vectorstore_utils import load_vector_store
from history import save_history
from performance_monitor import performance_monitor
from datetime import datetime
from config import TAVILY_API_KEY
import os
import time

# Define the state for LangGraph - removed add_messages annotation
class ChatState(TypedDict, total=False):
    query: str
    session_id: str
    answer: str
    performance: dict
    messages: list  # Changed: removed add_messages annotation
    vector_store: object
    docs: object

def load_vector_store_node(state: ChatState):
    start_time = time.time()
    session_id = state.get("session_id", "")
    vector_store = load_vector_store(session_id)
    vector_load_time = (time.time() - start_time) * 1000
    performance_monitor.metrics["vector_store_loading"].append(vector_load_time)
    print(f"[PERFORMANCE] Vector Store Loading: {vector_load_time:.2f}ms")
    return {"vector_store": vector_store, "performance": {"vector_store_loading_ms": vector_load_time}}

def similarity_search_node(state: ChatState):
    vector_store = state.get("vector_store")
    query = state.get("query", "")
    start_time = time.time()
    docs = vector_store.similarity_search(query)
    search_time = (time.time() - start_time) * 1000
    performance_monitor.metrics["similarity_search"].append(search_time)
    print(f"[PERFORMANCE] Similarity Search: {search_time:.2f}ms")
    return {"docs": docs, "performance": {**state.get("performance", {}), "similarity_search_ms": search_time}}

def llm_inference_node(state: ChatState):
    docs = state.get("docs")
    query = state.get("query", "")
    session_id = state.get("session_id", "")
    messages = state.get("messages", [])

    # Build context from conversation history
    context = build_context_from_history(messages, query)
    
    start_time = time.time()
    chain = get_chain(context_type="local")  # Use "local" for local context
    
    # Pass the built context to the chain
    response = chain({
        "input_documents": docs, 
        "question": query, 
        "context": context
    }, return_only_outputs=True)
    
    llm_time = (time.time() - start_time) * 1000
    performance_monitor.metrics["llm_inference"].append(llm_time)
    print(f"[PERFORMANCE] LLM Inference: {llm_time:.2f}ms")
    
    answer = response['output_text']
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf_names = "N/A"
    
    # Save the current conversation to history
    save_history(session_id, query, answer, "Google AI", timestamp, pdf_names)
    
    return {
        "answer": answer,
        "timestamp": timestamp,
        "performance": {**state.get("performance", {}), "llm_inference_ms": llm_time}
    }

def build_context_from_history(messages, current_query):
    """
    Build context string from conversation history.
    messages: list of dicts with 'question' and 'answer' keys
    current_query: the current user question
    """
    context = ""
    
    # Add previous conversations
    for msg in messages:
        if isinstance(msg, dict) and 'question' in msg and 'answer' in msg:
            context += f"User: {msg['question']}\nAssistant: {msg['answer']}\n\n"
    
    # Add current query
    context += f"User: {current_query}\nAssistant: "
    
    return context

def web_call_func(state: ChatState) -> str:
    answer = state.get("answer", "").lower()
    
    # Check if the answer indicates information is not available
    if any(phrase in answer for phrase in [
        "answer is not available", 
        "not available in the context",
        "cannot find information",
        "not mentioned in the context"
    ]):
        return "search"
    else:
        return "end"

from tavily import TavilyClient
from langchain.schema import Document

def tavily_call_func(state: ChatState) -> ChatState:
    query = state.get("query", "")
    session_id = state.get("session_id", "")
    
    try:
        # Create Tavily search tool
        tavily = TavilyClient(TAVILY_API_KEY)
        print(f"[TAVILY] Searching for: {query}")
        
        # Use Tavily to search the web
        results = tavily.search(query,exclude_domains=["https://en.wikipedia.org/wiki/"],max_results=3)
        print(f"[TAVILY] Search results: {results}")
        
        docs = []
        if results and 'results' in results:
            for result in results['results']:
                url = result.get('url', 'Unknown')
                content = result.get('content', 'No content')
                # Each web result becomes a Document with metadata for source (URL)
                docs.append(Document(page_content=content, metadata={"image_number": url, "image_file": url}))
            # Call the LLM with all web docs as input_documents
            chain = get_chain(context_type="web")
            # Use conversation history as context
            context = build_context_from_history(state.get("messages", []), query)
            response = chain({
                "input_documents": docs,
                "question": query,
                "context": context
            }, return_only_outputs=True)
            answer = response['output_text']
        else:
            answer = "No relevant information found from web search."
            
    except Exception as e:
        print(f"[TAVILY] Error during search: {e}")
        answer = f"Web search failed: {str(e)}"
    
    # Save the web search result to history
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_history(session_id, query, answer, "Tavily Web Search", timestamp, "Web Search")
    
    return {
        "query": query,
        "session_id": session_id,
        "answer": answer,
        "timestamp": timestamp
    }

# Build and export the LangGraph workflow
chat_graph = StateGraph(ChatState)
chat_graph.add_node("load_vector_store", load_vector_store_node)
chat_graph.add_node("similarity_search", similarity_search_node)
chat_graph.add_node("llm_inference", llm_inference_node)
chat_graph.add_node("tavily", tavily_call_func)

chat_graph.set_entry_point("load_vector_store")
chat_graph.add_edge("load_vector_store", "similarity_search")
chat_graph.add_edge("similarity_search", "llm_inference")
chat_graph.add_conditional_edges("llm_inference", web_call_func, {
    "search": "tavily",
    "end": END
})

compiled_chat_graph = chat_graph.compile()

async def run_chat_workflow_async(state: ChatState):
    """Async run for the chat workflow."""
    return await compiled_chat_graph.ainvoke(state)

def visualize_workflow_mermaid():
    """Return the Mermaid syntax for the workflow graph for visualization."""
    graph = compiled_chat_graph.get_graph()
    if hasattr(graph, 'draw_mermaid'):
        return graph.draw_mermaid()
    raise NotImplementedError("draw_mermaid is not available on the workflow graph.")