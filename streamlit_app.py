import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from controller import handle_user_query

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

if 'authenticator' not in st.session_state:
    st.session_state['authenticator'] = authenticator

authenticator.login('main')
 
# Check authentication status
if st.session_state.get("authentication_status"):
    if st.session_state["name"] == 'oracle':
        st.title("STUDio 📃")
        st.write("Welcome to STUDio, your one-stop solution for all your academic needs!")

        # Display chat messages from history
        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                if message["role"] == "user" and "files" in message:
                    if message["content"]:
                        st.markdown(message["content"])
                    for file in message["files"]:
                        st.image(file)
                else:
                    st.markdown(message["content"])

        # Accept user input with file upload
        user_input = st.chat_input(
            "Say something and/or attach a document",
            accept_file=True,
            file_type=["png", "docx", "pdf"],
        )

        if user_input:
            # Display user message
            with st.chat_message("user"):
                if user_input.text:
                    st.markdown(user_input.text)
                if user_input.files:
                    for file in user_input.files:
                        st.write("Attachment:", file.name)

            # Add user message to chat history
            st.session_state['messages'].append({
                "role": "user",
                "content": user_input.text,
                "files": user_input.files,
            })
            query = user_input.text
            if query:
                assistant_response = handle_user_query(query)
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

            # Add assistant response to chat history
            st.session_state['messages'].append({
                "role": "assistant",
                "content": assistant_response,
            })
        with st.sidebar:
            if authenticator.logout('Logout', 'main'):
                st.session_state.clear()
                st.write("You have logged out successfully!")
                st.stop()


# Run streamlit run streamlit run streamlit_app.py to start the app.
# You can now interact with the app in your browser.
# You can also log in and log out using the sidebar.