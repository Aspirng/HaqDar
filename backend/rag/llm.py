from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

def query_gemini(user_query: str, chunks: list) -> dict:
    """
    Generate an answer using OpenAI's gpt-4o-mini based on retrieved mock_chunks.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Format the retrieved chunks for the prompt
    context_text = ""
    sources = []
    for i, chunk in enumerate(chunks):
        law = chunk.get("law", "Unknown Law")
        section = chunk.get("section_ref", "Unknown Section")
        text = chunk.get("text", "")
        context_text += f"\n[Source {i+1}]: {law}, Section {section}\n{text}\n"
        sources.append({"law": law, "section": section})

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a highly professional and empathetic legal assistant specializing in Pakistani Law.\n"
            "You are bilingual and can communicate perfectly in both English and Urdu.\n\n"
            "GUIDELINES:\n"
            "1. Respond in the same language used by the user (if they ask in Urdu, respond in Urdu; if in English, respond in English).\n"
            "2. Use the provided context to answer the user's question accurately. The context is in English, so translate the information into Urdu if the user is asking in Urdu.\n"
            "3. If the provided context does not contain the answer, provide a detailed and polite apology in the user's language stating you don't have that specific information and advise consulting a professional.\n"
            "4. Keep the tone professional, helpful, and respectful.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Helpful Answer:"
        )
    )

    prompt = prompt_template.format(context=context_text, question=user_query)
    
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources
    }

if __name__ == "__main__":
    mock_chunks = [
        {
            "text": "Section 302 PPC — whoever commits murder shall be punished with death.",
            "law": "Pakistan Penal Code 1860", 
            "section_ref": "302", 
            "doc_type": "criminal"
        }
    ]
    # We changed the function name from query_gemini to query_llm since we are using OpenAI
    result = query_gemini("Someone attacked me with a knife", mock_chunks)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
