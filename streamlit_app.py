import torch
import streamlit as st
from peft import AutoPeftModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

@st.cache_resource
def get_model():
    original_model_name='google/flan-t5-base'
    my_model_name = 'BhushanM25/fine-tuned-flan-t5-base-v4'
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    
    my_model = AutoPeftModelForSeq2SeqLM.from_pretrained('BhushanM25/fine-tuned-flan-t5-base-v4',
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)
    return tokenizer, my_model

tokenizer, model = get_model()

def get_summary(conversation):
    prompt = f"""
Summarize the following conversation.

{conversation}

Summary:
"""

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    model_outputs = model.generate(input_ids=input_ids, 
                                   generation_config=GenerationConfig(max_new_tokens=200))
    text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    print("Generated fine-tuned output: ", text_output)
    return text_output

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = "user1"
    
# Create placeholders for chat messages
message_placeholders = {}

def convert_user_to_person(user):
    return "#Person1#" if user == "user1" else "#Person2#"

def toggle_user(current_user):
    # Toggle user logic
    if current_user == "user1":
        st.session_state['current_user'] = "user2"
    else:
        st.session_state['current_user'] = "user1"
        
def format_messages():
     # Collect messages from current session
    session_messages = st.session_state.get("messages")
    print("session_messages: ", session_messages)
    messages = []
    for message_dict in session_messages:
        person = "#Person1#" if message_dict["role"] == "user1" else "#Person2#"
        content = message_dict["content"]
        message = person + ": " + content + "\n"
        messages.append(message)
    return "".join(messages)


st.title("Conversation Summarization App")

# # Display chat messages from history on app rerun
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    # message_placeholders[role] = st.empty()
    with st.chat_message(role[-1]):
        st.markdown(content)

user_input = st.chat_input("Say something")

if user_input:
    
    current_user = st.session_state['current_user']
        
    st.session_state.messages.append({"role": current_user, "content": user_input})
    # Display user message in chat message container
    with st.chat_message(current_user[-1]):
        st.markdown(f'{current_user} says - {user_input}')
        
    toggle_user(current_user)
    

# Button to trigger summarization
if st.button("Summarize"):

    # Format messages as input prompt 
    # (Placeholder - you said you will handle this part)
    formatted_input = format_messages() 
    print("formatted_input: ", formatted_input)

    # Pass input to model for summarization
    summary_text = get_summary(formatted_input)  

    # Clear session history
    st.session_state.get("messages").clear()

    # Display summary  
    st.text_area("Summary:", f" Summary:\n{summary_text}", height=100)