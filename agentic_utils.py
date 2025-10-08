import os
import re
import json
import pandas as pd
import chromadb
from groq import Groq
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, TypedDict
import logging
from datetime import datetime
from dotenv import load_dotenv
from chroma_db_utils import get_or_create_collection

# Load environment variables
load_dotenv()

# Setup logging - single log file per session
log_dir = os.getenv('LOG_DIR', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/agent_flow.log"

# Clear previous handlers
logging.getLogger().handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize clients
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Initialize ChromaDB collection
try:
    collection = get_or_create_collection()
    logger.info("Successfully connected to ChromaDB collection")
except Exception as e:
    logger.error(f"Error connecting to ChromaDB: {e}")
    chroma_client = chromadb.PersistentClient(path=os.getenv('CHROMA_DB_PATH'))
    collection = chroma_client.create_collection("customer_service_kb")

class AgentState(TypedDict):
    audio_file: str
    transcript: str
    extracted_info: Dict[str, Any]
    retrieved_context: List[Dict]
    generated_response: str
    final_output: Dict[str, Any]
    agent_logs: List[Dict]

def log_agent_step(agent_name: str, status: str, message: str, state: AgentState):
    """Log agent step and store in state"""
    log_entry = {
        "agent": agent_name,
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if "agent_logs" not in state:
        state["agent_logs"] = []
    state["agent_logs"].append(log_entry)
    
    if status == "success":
        logger.info(f"===== {agent_name} =====\n{message}\n")
    else:
        logger.error(f"===== {agent_name} =====\n{message}\n")
    
    return state

def transcription_agent(state: AgentState) -> AgentState:
    """Agent 1: Convert speech to text"""
    try:
        with open(state["audio_file"], "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(state["audio_file"], file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        
        state["transcript"] = transcription.text
        return log_agent_step(
            "Transcription Agent", 
            "success", 
            f"Transcript: {state['transcript']}", 
            state
        )
    except Exception as e:
        return log_agent_step(
            "Transcription Agent", 
            "error", 
            f"Error: {str(e)}", 
            state
        )

def info_extractor_agent(state: AgentState) -> AgentState:
    """Agent 2: Extract topic, description, and sentiment"""
    try:
        prompt = f"""Analyze this customer conversation from telecom domain and extract the following information in JSON format:

Conversation: {state['transcript']}

Extract and return ONLY a valid JSON object with these exact keys:
- "topic_name": main topic or issue
- "description": brief description of the user query in 1 to 2 sentences
- "overall_sentiment": positive/negative/neutral

Return ONLY the JSON object, no additional text or explanation.
"""
        
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Clean the response - remove any markdown formatting
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Try to parse JSON directly
        try:
            extracted_data = json.loads(response_text)
            state["extracted_info"] = extracted_data
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0).strip()
                extracted_data = json.loads(json_str)
                state["extracted_info"] = extracted_data
            else:
                raise ValueError("No valid JSON found in response")
        
        return log_agent_step(
            "Info Extractor Agent", 
            "success", 
            f"Extracted Info: {json.dumps(state['extracted_info'], indent=2)}", 
            state
        )
    except Exception as e:
        state["extracted_info"] = {
            "topic_name": "general_inquiry",
            "description": state["transcript"][:100] + "...",
            "overall_sentiment": "neutral"
        }
        return log_agent_step(
            "Info Extractor Agent", 
            "error", 
            f"Error: {str(e)} - Using fallback data", 
            state
        )

def context_retrieval_agent(state: AgentState) -> AgentState:
    """Agent 3: Retrieve context and generate response"""
    try:
        # Retrieve similar documents
        query_text = f"{state['extracted_info']['topic_name']} {state['extracted_info']['description']}"
        
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=3,
                include=["documents", "metadatas"]
            )
            
            state["retrieved_context"] = []
            for i, doc in enumerate(results['documents'][0]):
                state["retrieved_context"].append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i]
                })
        except Exception as e:
            logger.warning(f"ChromaDB query failed: {e}")
            state["retrieved_context"] = []
        
        # Generate response
        context_str = "\n".join([f"• {doc['content']}" for j, doc in enumerate(state["retrieved_context"])]) if state["retrieved_context"] else "No specific solutions found in knowledge base"

        prompt = f"""You are a telecom customer support agent. Generate a helpful response to the customer.

CUSTOMER CONVERSATION:
{state['transcript']}

CUSTOMER SENTIMENT: {state['extracted_info']['overall_sentiment']}

AVAILABLE CONTEXT:
{context_str}

IMPORTANT INSTRUCTIONS:
1. Start directly with a business-friendly response - no technical openings like "Based on results" or "Here's a possible response"
2. Be concise and helpful (4-5 sentences maximum)
3. Use bullet points if listing multiple items
4. If the customer needs follow-up or detailed assistance that can't be resolved here, tell them to email their details to: support@gmail.com
5. Do NOT ask "Can you provide..." or "Could you give..." - directly instruct them to email if needed
6. Use the available context if relevant, otherwise use your knowledge to provide the best solution
7. If the query is NOT related to telecom services OR contains harmful/illegal requests OR asks for passwords, respond with: "I'm sorry, but I can only assist with telecom-related queries. Please ask questions related to our telecom services."

Generate the response:
"""
        
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        state["generated_response"] = completion.choices[0].message.content.strip()
        
        return log_agent_step(
            "Context Retrieval Agent", 
            "success", 
            f"{state['generated_response']}", 
            state
        )
    except Exception as e:
        return log_agent_step(
            "Context Retrieval Agent", 
            "error", 
            f"Error: {str(e)}", 
            state
        )

def create_workflow():
    """Create and return the complete agent workflow - ONLY processing, no approval/update"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("transcribe", transcription_agent)
    workflow.add_node("extract", info_extractor_agent)
    workflow.add_node("retrieve", context_retrieval_agent)
    
    workflow.set_entry_point("transcribe")
    workflow.add_edge("transcribe", "extract")
    workflow.add_edge("extract", "retrieve")
    workflow.add_edge("retrieve", END)
    
    return workflow.compile()

def run_agent_flow(audio_file_path: str):
    """Execute the complete agent workflow - ONLY processing"""
    workflow = create_workflow()
    
    initial_state = AgentState(
        audio_file=audio_file_path,
        transcript="",
        extracted_info={},
        retrieved_context=[],
        generated_response="",
        final_output={},
        agent_logs=[]
    )
    
    logger.info("🚀 Starting Multi-Agent Workflow...")
    result = workflow.invoke(initial_state)
    
    # Prepare final output without approval/update
    result["final_output"] = {
        "transcript": result["transcript"],
        "extracted_info": result["extracted_info"],
        "retrieved_solutions": len(result["retrieved_context"]),
        "generated_response": result["generated_response"],
        "requires_human_approval": True  # Flag for support engineer
    }
    
    logger.info("🎯 Processing Completed - Ready for Human Review!")
    
    return result