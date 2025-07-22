from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from pydantic import SecretStr
from config import GOOGLE_API_KEY

def get_chain():
    prompt_template = """
You are an intelligent assistant with access to document content and conversation history. 

**Instructions:**
1. First, check the conversation history to see if this topic has been discussed before
2. Then, use the provided document context to answer the question
3. If the answer is available in either the conversation history or document context, provide a comprehensive response
4. If the answer is NOT available in either source, respond with "Answer is not available in the context"
5. Always be accurate and don't make up information
6. At the end of your response, always mention the URL or  page number or image number of the source(s) you used at the last with format "Source:[source-name]"
7. At the end of your response, also list the bounding boxes and source filenames of the image chunks you used, in the format: [Source: <filename>, BBox: [x_min, y_min, x_max, y_max]]

**Document Context:**
{context}

**Conversation History & Current Question:**
{question}

**Your Response:**
"""
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", 
        temperature=0.3, 
        google_api_key=SecretStr(GOOGLE_API_KEY)
    )
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    def chain(inputs, return_only_outputs=False):
        # Extract document context from input_documents
        doc_context = ""
        if "input_documents" in inputs and inputs["input_documents"]:
            doc_context = "\n".join([
                f"[Image Number: {getattr(doc, 'metadata', {}).get('image_number', 'N/A')}, File: {getattr(doc, 'metadata', {}).get('image_file', 'N/A')}] {doc.page_content}"
                for doc in inputs["input_documents"]
            ])
        # Get conversation context and current question
        conversation_context = inputs.get("context", "")
        current_question = inputs.get("question", "")
        # Format the final prompt
        formatted_prompt = prompt_template.format(
            context=doc_context,
            question=conversation_context
        )
        print(f"[DEBUG] Formatted prompt: {formatted_prompt[:500]}...")
        # Call the LLM
        try:
            output_text = model.predict(formatted_prompt)
            print(f"[DEBUG] LLM response: {output_text}...")
            return {"output_text": output_text}
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            return {"output_text": f"Error generating response: {str(e)}"}
    
    return chain