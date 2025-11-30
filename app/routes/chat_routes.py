# app/routes/chat_routes.py
import os
from typing import List
from fastapi import APIRouter, Request, HTTPException, status
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
import google.generativeai as genai

from app.config import logger, vector_store
from app.models import ChatRequest, ChatResponse, SourceDocument
from app.services.vector_store.async_pg_vector import AsyncPgVector

router = APIRouter()


def get_azure_client():
    """Initialize Azure OpenAI client for chat completions."""
    endpoint = os.getenv("AZURE_CHAT_ENDPOINT", "https://ai-40mini.cognitiveservices.azure.com/")
    api_key = os.getenv("AZURE_CHAT_API_KEY", "")

    if not api_key:
        raise ValueError("AZURE_CHAT_API_KEY environment variable not set")

    return AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint=endpoint,
        api_key=api_key
    )


def get_gemini_client(model_name: str = 'gemini-2.5-flash'):
    """Initialize Google Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def format_sources_for_context(sources: List[tuple]) -> str:
    """Format retrieved sources into a context string for the LLM."""
    context_parts = []
    for idx, (doc, score) in enumerate(sources, 1):
        context_parts.append(f"[Source {idx}] (Relevance: {score:.3f})\n{doc.page_content}\n")
    return "\n".join(context_parts)


def create_rag_prompt(query: str, context: str) -> str:
    """Create a RAG prompt that instructs the LLM to answer based on context."""
    return f"""You are a helpful AI assistant that answers questions based on the provided document context.

IMPORTANT INSTRUCTIONS:
1. Answer the question using ONLY the information from the provided sources below
2. If the answer cannot be found in the sources, say "I cannot find this information in the provided document"
3. Be specific and cite which source number you're using when possible
4. If sources contradict each other, mention both perspectives
5. Keep your answer concise but complete

SOURCES:
{context}

QUESTION:
{query}

ANSWER:"""


def create_chat_messages(query: str, context: str) -> List[dict]:
    """Create messages for chat completion API."""
    system_prompt = """You are a helpful AI assistant specialized in answering questions about documents.
You must ONLY use the information provided in the sources to answer questions.
If the information is not in the sources, clearly state that you cannot answer based on the provided documents.
Be accurate, concise, and always cite your sources when possible."""

    user_prompt = f"""Based on the following sources from the document, please answer the question.

SOURCES:
{context}

QUESTION: {query}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


async def retrieve_relevant_documents(
    query: str,
    file_id: str,
    k: int,
    request: Request
) -> List[tuple]:
    """Retrieve relevant documents from vector store."""
    try:
        # Get embedding for the query
        embedding = vector_store.embedding_function.embed_query(query)

        # Search for similar documents
        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=k,
                filter={"file_id": file_id},
                executor=request.app.state.thread_pool,
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=k, filter={"file_id": file_id}
            )

        return documents
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}"
        )


async def generate_azure_response(messages: List[dict], temperature: float) -> str:
    """Generate response using Azure OpenAI."""
    try:
        client = get_azure_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Azure OpenAI error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Azure OpenAI error: {str(e)}"
        )


async def generate_gemini_response(prompt: str, temperature: float, model_name: str = 'gemini-2.5-flash') -> str:
    """Generate response using Google Gemini."""
    try:
        model = get_gemini_client(model_name)
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": 1000,
        }
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Check if response has text (not blocked by safety filters)
        try:
            return response.text
        except ValueError as e:
            # Response blocked by safety filters or other content policy
            finish_reason = getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None
            if finish_reason == 2:  # SAFETY
                return "I cannot provide a response to this query due to content safety policies."
            elif finish_reason == 3:  # RECITATION
                return "I cannot provide a response as it may contain recited content."
            else:
                return "I cannot generate a response for this query."

    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini API error: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: Request, body: ChatRequest):
    """
    Chat endpoint that retrieves relevant documents and generates AI responses.

    Supports:
    - Azure OpenAI GPT-4o-mini
    - Google Gemini
    """
    try:
        # Retrieve relevant documents
        logger.info(f"Retrieving documents for query: {body.query[:50]}...")
        documents = await retrieve_relevant_documents(
            body.query,
            body.file_id,
            body.k,
            request
        )

        if not documents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found for the query"
            )

        # Format context from retrieved documents
        context = format_sources_for_context(documents)
        logger.info(f"Retrieved {len(documents)} relevant documents")

        # Generate response based on selected model
        if body.model == "azure-gpt4o-mini":
            messages = create_chat_messages(body.query, context)
            answer = await generate_azure_response(messages, body.temperature)
            model_used = "Azure GPT-4o-mini"
        elif body.model in ["gemini", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
            # Map generic "gemini" to the default model
            actual_model = "gemini-2.5-flash" if body.model == "gemini" else body.model
            prompt = create_rag_prompt(body.query, context)
            answer = await generate_gemini_response(prompt, body.temperature, actual_model)
            model_used = f"Google Gemini ({actual_model})"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported model: {body.model}"
            )

        # Prepare sources for response
        sources = [
            SourceDocument(
                content=doc.page_content,
                score=float(score),
                metadata=doc.metadata
            )
            for doc, score in documents
        ]

        logger.info(f"Generated response using {model_used}")

        return ChatResponse(
            answer=answer,
            sources=sources,
            model_used=model_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
