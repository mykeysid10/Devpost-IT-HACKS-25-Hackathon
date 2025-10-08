import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
from agentic_utils import run_agent_flow
from chroma_db_utils import get_next_id, add_to_chroma_only, get_or_create_collection
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Customer Support Agentic AI Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean hot color scheme CSS - Red, Orange, Yellow with Green accents
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #DC2626;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .agent-success {
        background: #FEF2F2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #16A34A;
        margin: 0.5rem 0;
    }
    .agent-error {
        background: #FEF2F2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #DC2626;
        margin: 0.5rem 0;
    }
    .response-box {
        background: #FEF2F2;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #EA580C;
        margin: 1rem 0;
    }
    .stTextArea textarea {
        background-color: white !important;
        color: #1f2937 !important;
        cursor: not-allowed !important;
        border: 1px solid #d1d5db !important;
        font-weight: 500 !important;
    }
    .success-badge {
        background-color: #16A34A;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .error-badge {
        background-color: #DC2626;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #6b7280;
        border-top: 1px solid #e5e7eb;
    }
    .bullet-points {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #EA580C;
    }
    .user-profile {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .small-text {
        font-size: 0.8rem;
        color: #6b7280;
    }
    .stats-container {
        background: #FEF2F2;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    if 'agent_results' not in st.session_state:
        st.session_state.agent_results = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "About"
    if 'form_reset' not in st.session_state:
        st.session_state.form_reset = False
    if 'last_page' not in st.session_state:
        st.session_state.last_page = "About"
    if 'user_role' not in st.session_state:
        st.session_state.user_role = "Customer"
    if 'pending_approvals' not in st.session_state:
        st.session_state.pending_approvals = []
    if 'approved_cases' not in st.session_state:
        st.session_state.approved_cases = []
    if 'customer_upload_status' not in st.session_state:
        st.session_state.customer_upload_status = "idle"  # idle, uploaded, processing, approved
    if 'approval_completed' not in st.session_state:
        st.session_state.approval_completed = False
    if 'go_to_dashboard' not in st.session_state:
        st.session_state.go_to_dashboard = False

def show_user_selection():
    st.sidebar.markdown("### 👤 Select User Role")
    user_role = st.sidebar.radio("Choose your role:", ["Customer", "Support Engineer"])
    st.session_state.user_role = user_role
    return user_role

def show_about():
    st.markdown('<div class="main-header">Customer Support - AI Agentic Framework</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 About
        
        This AI-powered customer support system revolutionizes customer service with multi-agent intelligence:
        
        - 🎙️ **Real-time Call Transcription** - Convert customer calls to text instantly
        - 😊 **Sentiment Analysis** - Understand customer emotions and urgency  
        - 🔍 **Smart Information Extraction** - Identify key issues and topics
        - 📚 **Knowledge Retrieval** - Find proven solutions from past cases
        - 💬 **Empathetic Response Generation** - Create human-like responses
        - ✅ **Human-in-the-Loop Validation** - Ensure quality before sending
        - 🧠 **Continuous Learning** - Improve with every interaction
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Tech Stack
        
        - **Groq API** - Whisper + LLaMA for lightning-fast inference
        - **LangGraph** - Advanced agent orchestration  
        - **ChromaDB** - Vector database for semantic search
        - **Streamlit** - Interactive UI
        - **Python** - Backend processing and logic
        """)
    
    # Footer
    st.markdown('<div class="footer">Built by: Siddharth Kulkarni</div>', unsafe_allow_html=True)


def show_architecture():
    st.markdown("## System Architecture")
    st.markdown("---")
    
    # Load and display the architecture image
    if os.path.exists("system_architecture.png"):
        st.image("system_architecture.png", use_container_width=True)
        st.markdown("---")
    else:
        st.info("Architecture diagram image not found. Please add 'system_architecture.png' to view the system architecture.")
    
    st.markdown("### Agent Workflow Steps")
    
    # Clean horizontal layout for workflow steps
    workflow_steps = [
        {"icon": "🎙️", "title": "Transcription Agent", "desc": "Converts speech to text using Whisper with 95%+ accuracy"},
        {"icon": "😊", "title": "Sentiment Analysis", "desc": "Analyzes emotional tone and customer satisfaction level"},
        {"icon": "🔍", "title": "Info Extraction", "desc": "Identifies key topics, issues, and customer intent"},
        {"icon": "📚", "title": "Context Retrieval", "desc": "Finds similar past solutions using semantic search"},
        {"icon": "💬", "title": "Response Genration", "desc": "Creates empathetic, context-aware responses"}
    ]
    
    # Create clean horizontal columns for steps
    cols = st.columns(len(workflow_steps))
    
    for i, (col, step) in enumerate(zip(cols, workflow_steps)):
        with col:
            # Simple centered content without borders or background
            st.markdown(f"<div style='text-align: center; padding: 1rem;'>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='font-size: 2rem; margin: 0;'>{step['icon']}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='margin: 0.5rem 0;'>{step['title']}</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #666; font-size: 0.9rem;'>{step['desc']}</p>", unsafe_allow_html=True)
            st.markdown(f"</div>", unsafe_allow_html=True)

def show_customer_support_flow():
    # Check if we need to go to dashboard (show empty state)
    if st.session_state.go_to_dashboard:
        st.session_state.go_to_dashboard = False
        st.session_state.agent_results = None
        st.session_state.approval_completed = False
    
    # Clear results when navigating to this page
    if st.session_state.last_page != "Customer Support":
        st.session_state.agent_results = None
        st.session_state.approval_completed = False
    st.session_state.last_page = "Customer Support"
    
    # Show empty state if no pending cases for Support Engineer
    if (st.session_state.user_role == "Support Engineer" and 
        not st.session_state.agent_results and 
        not st.session_state.pending_approvals):
        
        st.markdown('<div class="main-header">Support Engineer Dashboard</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        st.info("🎯 **Active Domain: Telecom** - Mobile plans, internet services, billing issues, network problems")
        
        st.markdown("---")
        
        st.markdown("### 👨‍💼 Support Engineer View - Review pending cases and manage knowledge base")
        
        if st.session_state.approved_cases:
            st.success(f"✅ You have {len(st.session_state.approved_cases)} approved cases in the knowledge base")
        
        st.info("📋 **No pending cases to review at the moment.**")
        st.markdown("When customers submit new cases, they will appear here for your review and approval.")
        
        # Footer
        st.markdown('<div class="footer">Built by: Siddharth Kulkarni</div>', unsafe_allow_html=True)
        return
    
    # Regular customer support flow for customers or when there are pending cases
    st.markdown('<div class="main-header">Customer Support Automation</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Telecom domain is fixed
    st.info("🎯 **Active Domain: Telecom** - Mobile plans, internet services, billing issues, network problems")
    
    st.markdown("---")
    
    # Sample audio files section for Customer
    if st.session_state.user_role == "Customer":
        st.markdown("### 🎵 Sample Audio Files")
        sample_dir = "sample_audio"
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.m4a', '.mp3', '.wav', '.ogg'))]
            if sample_files:
                for sample_file in sample_files[:2]:  # Show only first 2 samples
                    sample_path = os.path.join(sample_dir, sample_file)
                    with open(sample_path, "rb") as file:
                        st.download_button(
                            label=f"📥 Download {sample_file}",
                            data=file,
                            file_name=sample_file,
                            mime="audio/mpeg",
                            key=f"sample_{sample_file}"
                        )
        else:
            st.info("No sample audio files found in 'sample_audio' directory")
        
        st.markdown("---")
    
    # File upload section - Only for Customer
    if st.session_state.user_role == "Customer":
        st.markdown("### 🎙️ Upload Customer Call")
        
        uploaded_file = st.file_uploader(
            "Upload Customer Call Audio", 
            type=['m4a', '.mp3', '.wav', '.ogg'],
            help="Supported formats: M4A, MP3, WAV, OGG"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            
            st.json(file_details)
            
            # Save uploaded file
            audio_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("🚀 Start AI Agent Flow", type="primary", use_container_width=True):
                with st.spinner("🤖 Processing audio through AI agents..."):
                    try:
                        result = run_agent_flow(audio_path)
                        st.session_state.agent_results = result
                        
                        # Store for support engineer approval
                        st.session_state.pending_approvals.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'filename': uploaded_file.name,
                            'result': result,
                            'case_id': len(st.session_state.pending_approvals) + 1,
                            'status': 'pending'
                        })
                        
                        # Set customer status
                        st.session_state.customer_upload_status = "uploaded"
                            
                    except Exception as e:
                        st.error(f"Error running agent flow: {str(e)}")
                
                # Clean up
                if os.path.exists(audio_path):
                    os.remove(audio_path)
    else:
        # For Support Engineer, show message when there are pending cases
        if st.session_state.pending_approvals:
            st.info(f"👨‍💼 **Support Engineer View** - You have {len(st.session_state.pending_approvals)} pending case(s) to review")
    
    # Show results if available
    if st.session_state.agent_results:
        if st.session_state.user_role == "Customer":
            display_customer_results(st.session_state.agent_results)
        else:
            display_support_engineer_results(st.session_state.agent_results)
    
    # Footer
    st.markdown('<div class="footer">Built by: Siddharth Kulkarni</div>', unsafe_allow_html=True)

def display_customer_results(result):
    """Display results for Customer role"""
    # Check if this case has been approved
    current_case_approved = any(
        case['result']['final_output']['extracted_info'] == result['final_output']['extracted_info']
        for case in st.session_state.approved_cases
    )
    
    # Update customer status
    if current_case_approved:
        st.session_state.customer_upload_status = "approved"
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    total_agents = len(result.get('agent_logs', []))
    successful_agents = len([log for log in result.get('agent_logs', []) if log['status'] == "success"])
    
    with col1:
        st.metric("Total Agents", total_agents)
    with col2:
        st.metric("Successful", successful_agents)
    with col3:
        if current_case_approved:
            status = "✅ Approved"
            st.session_state.customer_upload_status = "approved"
        else:
            status = "⏳ Pending Approval"
        st.metric("Status", status)
    
    st.markdown("---")
    
    # Display agent logs
    st.markdown("### 🔍 Detailed Agent Execution")
    
    for i, log in enumerate(result.get('agent_logs', [])):
        agent_name = log['agent']
        status = log['status']
        message = log['message']
        
        if status == "success":
            st.markdown(f"""
            <div class="agent-success">
                <h4>✅ {agent_name} - <span class="success-badge">SUCCESS</span></h4>
            </div>
            """, unsafe_allow_html=True)
            st.text_area(
                f"{agent_name} Output", 
                message, 
                height=150, 
                key=f"success_{agent_name}_{i}",
                disabled=True
            )
        else:
            st.markdown(f"""
            <div class="agent-error">
                <h4>❌ {agent_name} - <span class="error-badge">ERROR</span></h4>
            </div>
            """, unsafe_allow_html=True)
            st.text_area(
                f"{agent_name} Error", 
                message, 
                height=100, 
                key=f"error_{agent_name}_{i}",
                disabled=True
            )
        
        st.markdown("---")
    
    # Final response section
    response_data = result['final_output']['generated_response']
    
    if isinstance(response_data, str):
        # Create downloadable text file
        solution_content = response_data
        st.download_button(
            label="📥 Download Solution as Text File",
            data=solution_content,
            file_name="solution.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Display response in the same style as other agents
        st.markdown(f"""
        <div class="agent-success">
            <h4>✅ Recommended Response - <span class="success-badge">SUCCESS</span></h4>
        </div>
        """, unsafe_allow_html=True)
        st.text_area(
            "Recommended Response", 
            response_data, 
            height=200, 
            key="recommended_response",
            disabled=True
        )
    
    st.markdown("---")
    
    # Show status message
    if current_case_approved:
        st.success("✅ Your case has been approved by Support Engineer and added to knowledge base!")
    else:
        st.info("🕒 Your case is pending approval from Support Engineer. It will be added to knowledge base once approved.")
    
    # Only show process another call button
    if st.button("🔄 Process Another Call", use_container_width=True):
        st.session_state.agent_results = None
        st.session_state.customer_upload_status = "idle"
        st.rerun()

def display_support_engineer_results(result):
    """Display results for Support Engineer role with editing and approval options"""
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    total_agents = len(result.get('agent_logs', []))
    successful_agents = len([log for log in result.get('agent_logs', []) if log['status'] == "success"])
    
    with col1:
        st.metric("Total Agents", total_agents)
    with col2:
        st.metric("Successful", successful_agents)
    with col3:
        st.metric("Status", "⏳ Needs Review")
    
    st.markdown("---")
    
    # Display agent logs
    st.markdown("### 🔍 AI Agent Execution Details")
    
    for i, log in enumerate(result.get('agent_logs', [])):
        agent_name = log['agent']
        status = log['status']
        message = log['message']
        
        if status == "success":
            st.markdown(f"""
            <div class="agent-success">
                <h4>✅ {agent_name} - <span class="success-badge">SUCCESS</span></h4>
            </div>
            """, unsafe_allow_html=True)
            st.text_area(
                f"{agent_name} Output", 
                message, 
                height=150, 
                key=f"success_{agent_name}_{i}",
                disabled=True
            )
        else:
            st.markdown(f"""
            <div class="agent-error">
                <h4>❌ {agent_name} - <span class="error-badge">ERROR</span></h4>
            </div>
            """, unsafe_allow_html=True)
            st.text_area(
                f"{agent_name} Error", 
                message, 
                height=100, 
                key=f"error_{agent_name}_{i}",
                disabled=True
            )
        
        st.markdown("---")
    
    # Approval section for Support Engineer with editing capability
    
    st.markdown("### ✅ Review & Add to Knowledge Base")
    st.markdown("---")
    
    extracted_info = result['final_output']['extracted_info']
    initial_response = result['final_output']['generated_response']
    
    # Editable response
    st.markdown("#### 📝 AI Generated Response (Editable)")
    edited_response = st.text_area(
        "Edit the response if needed:",
        value=initial_response,
        height=200,
        key="editable_response"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Approve & Add to Knowledge Base", type="primary", use_container_width=True):
            # Use edited response if changes were made
            final_response = edited_response if edited_response != initial_response else initial_response
            
            new_id = get_next_id()
            success = add_to_chroma_only(
                case_id=new_id,
                topic_name=extracted_info['topic_name'],
                description=extracted_info['description'],
                sentiment=extracted_info['overall_sentiment'],
                solution=final_response
            )
            
            if success:
                # Add to approved cases
                st.session_state.approved_cases.append({
                    'result': result,
                    'approved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                # Remove from pending approvals
                if st.session_state.pending_approvals:
                    st.session_state.pending_approvals = [
                        case for case in st.session_state.pending_approvals 
                        if case['result'] != result
                    ]
                # Update customer status
                st.session_state.customer_upload_status = "approved"
                # Set approval completed flag
                st.session_state.approval_completed = True
                # Set flag to go to dashboard
                st.session_state.go_to_dashboard = True
                
                st.success("✅ Case approved and added to knowledge base successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to add to knowledge base")
    
    with col2:
        if st.button("❌ Reject Response", use_container_width=True):
            st.warning("Response rejected and will not be added to knowledge base")
            # Remove from pending approvals
            if st.session_state.pending_approvals:
                st.session_state.pending_approvals = [
                    case for case in st.session_state.pending_approvals 
                    if case['result'] != result
                ]
            # Set approval completed flag
            st.session_state.approval_completed = True
            # Set flag to go to dashboard
            st.session_state.go_to_dashboard = True
            st.rerun()

def show_knowledge_base():
    st.markdown('<div class="main-header">Knowledge Base Management</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Side-by-side layout for Issue 1
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Knowledge Base Stats")
        st.markdown("---")
        
        # Get collection info
        try:
            collection = get_or_create_collection()
            count = collection.count()
            
            # Display stats in a compact format - REMOVED the pink box background
            col1a, col2a, col3a = st.columns(3)
            with col1a:
                st.metric("Total Cases", count, label_visibility="visible")
            with col2a:
                st.metric("Pending", len(st.session_state.pending_approvals), label_visibility="visible")
            with col3a:
                st.metric("Approved", len(st.session_state.approved_cases), label_visibility="visible")
            
            # Show ChromaDB records in a table
            st.markdown("---")
            st.markdown("### 📋 ChromaDB Records")
            st.markdown("---")
            st.markdown('<div class="small-text">Live view of cases in vector database</div>', unsafe_allow_html=True)
            
            try:
                # Get all records from ChromaDB
                results = collection.get()
                if results['ids']:
                    # Create a DataFrame from the metadata
                    records_data = []
                    for i, metadata in enumerate(results['metadatas']):
                        records_data.append({
                            'ID': metadata.get('id', 'N/A'),
                            'Topic': metadata.get('topic_name', 'N/A'),
                            'Sentiment': metadata.get('sentiment', 'N/A'),
                            'Source': metadata.get('source', 'N/A')
                        })
                    
                    df_records = pd.DataFrame(records_data)
                    st.dataframe(df_records, use_container_width=True, height=200)
                    
                    # Show record count
                    st.markdown(f'<div class="small-text">Showing {len(df_records)} records from ChromaDB</div>', unsafe_allow_html=True)
                else:
                    st.info("No records found in ChromaDB")
                    
            except Exception as e:
                st.error(f"Error fetching ChromaDB records: {e}")
            
        except Exception as e:
            st.error(f"Error accessing knowledge base: {e}")
        
        #st.markdown("---")
    
    with col2:
        st.markdown("### ➕ Add New Case to Knowledge Base")
        st.markdown("---")
        
        # Get next ID automatically
        next_id = get_next_id()
        st.text_input("Case ID", value=str(next_id), disabled=True, help="Automatically generated ID")
        
        # Use session state to manage form fields for reset
        if 'form_reset' not in st.session_state:
            st.session_state.form_reset = False
            
        if st.session_state.form_reset:
            topic = st.text_input("Topic Name *", placeholder="e.g., Billing Issue, Network Problem", key="reset_topic")
            description = st.text_area("Description *", placeholder="Detailed description of the customer issue...", height=100, key="reset_description")
            sentiment = st.selectbox("Customer Sentiment *", ["positive", "negative", "neutral"], key="reset_sentiment")
            solution = st.text_area("Solution *", placeholder="Proven solution for this issue...", height=150, key="reset_solution")
        else:
            topic = st.text_input("Topic Name *", placeholder="e.g., Billing Issue, Network Problem", key="topic")
            description = st.text_area("Description *", placeholder="Detailed description of the customer issue...", height=100, key="description")
            sentiment = st.selectbox("Customer Sentiment *", ["positive", "negative", "neutral"], key="sentiment")
            solution = st.text_area("Solution *", placeholder="Proven solution for this issue...", height=150, key="solution")
        
        if st.button("💾 Add New Case to Knowledge Base", type="primary", use_container_width=True):
            if topic and description and solution:
                # Show indexing progress
                with st.spinner("🔍 Indexing in knowledge base..."):
                    success = add_to_chroma_only(
                        case_id=next_id,
                        topic_name=topic,
                        description=description,
                        sentiment=sentiment,
                        solution=solution
                    )
                    
                    time.sleep(1)  # Simulate indexing time
                
                if success:
                    st.success("✅ Case successfully indexed in knowledge base!")
                    # Reset form by toggling the reset state
                    st.session_state.form_reset = not st.session_state.form_reset
                    st.rerun()
                else:
                    st.error("❌ Failed to add case to knowledge base")
            else:
                st.error("❌ Please fill all required fields (marked with *)")
    
    # Footer
    st.markdown('<div class="footer">Built by: Siddharth Kulkarni</div>', unsafe_allow_html=True)

def main():
    init_session_state()
    
    # Sidebar - Always show About and Architecture first
    st.sidebar.title("🧭 Navigation")
    # FIXED: Only one dash here instead of two
    st.sidebar.markdown("---")
    
    # Always show these pages
    pages = {
        "🎯 About": show_about,
        "🏗️ Architecture": show_architecture,
    }
    
    # User role selection
    user_role = show_user_selection()
    
    # FIXED: Only one dash here instead of two
    st.sidebar.markdown("---")
    
    # Add role-specific pages
    if user_role == "Customer":
        pages["🎙️ Customer Support"] = show_customer_support_flow
        # Show customer status in sidebar
        if st.session_state.customer_upload_status != "idle":
            # st.sidebar.markdown("---")
            status_text = {
                "uploaded": "⏳ Case Pending Approval",
                "approved": "✅ Case Approved"
            }.get(st.session_state.customer_upload_status, "📝 Ready to Upload")
            # st.sidebar.markdown(f"### {status_text}")
    else:  # Support Engineer
        pages["🎙️ Customer Support"] = show_customer_support_flow
        pages["📚 Knowledge Base"] = show_knowledge_base
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Display the selected page
    pages[selection]()

if __name__ == "__main__":
    main()