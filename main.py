import streamlit as st
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag import Embeddings
from chatbot import Chatbot_class
import os
from tts import TTS_Generator



def display1(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'medical_model' not in st.session_state:
    st.session_state['medical_model'] = None

if 'tts_generator' not in st.session_state:
    st.session_state['tts_generator'] = TTS_Generator()  # Initialize TTSGenerator

model_dir = "./models/Gemma-2-2b-it-ChatDoctor"


def load_medical_model():
    if not os.path.exists(model_dir):
        with st.spinner("Downloading and saving model..."):
            try:
                from huggingface_hub import login
                login("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")
                model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
                model.save_pretrained(model_dir)
                st.success("Model downloaded and saved locally!")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
    else:
        with st.spinner("Loading model from local directory..."):
            try:
                model = AutoModelForCausalLM.from_pretrained(model_dir)
                st.success("Model loaded from local directory!")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
    return model


st.set_page_config(
    page_title="Ustudy RAG Application",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.markdown("Your personal PDF analyser")
    menu = ['ChatBot RAG', 'MedicalBot', 'TTS', 'Voice Cloning']
    choice = st.selectbox("Models:", menu)

if choice == "ChatBot RAG":
    st.title("Llama 3 RAG")
    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            st.success("File Uploaded Successfully!")
            st.markdown(f"Filename: {uploaded_file.name}")
            st.markdown(f"File Size: {uploaded_file.size} bytes")

            create_embeddings = st.checkbox("Create Embeddings")
            if create_embeddings:
                if st.session_state['temp_pdf_path'] is None:
                    st.warning("Please upload a PDF first")
                else:
                    try:
                        embeddings_manager = Embeddings(
                            model_name="BAAI/bge-small-en",
                            device="cpu",
                            encode_kwargs={"normalize_embeddings": True},
                            qdrant_url="http://localhost:6333",
                            connection_name="vector_db"
                        )

                        with st.spinner("Creating embeddings..."):
                            result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                        st.success(result)

                        if st.session_state['chatbot_manager'] is None:
                            st.session_state['chatbot_manager'] = Chatbot_class(
                                model_name="BAAI/bge-small-en",
                                device="cpu",
                                encode_kwargs={"normalize_embeddings": True},
                                llm_model="llama3",
                                temperature=0.7,
                                qdrant_url="http://localhost:6333",
                                collection_name="vector_db"
                            )


                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

            st.markdown("Preview")
            display1(uploaded_file)

            temp_pdf_path = "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state['temp_pdf_path'] = temp_pdf_path

    with col2:
        st.header("Chat with PDF")

        if st.session_state['chatbot_manager'] is None:
            st.info("Please upload a PDF and create embeddings to start chatting.")
        else:
            for msg in st.session_state['messages']:
                st.chat_message(msg['role']).markdown(msg['content'])
            if user_input := st.chat_input("Type your message here..."):
                st.chat_message("user").markdown(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})

                with st.spinner("Answering..."):
                    try:
                        answer = st.session_state['chatbot_manager'].get_response(user_input)
                    except Exception as e:
                        answer = f"An error occurred while processing request: {e}"

                st.chat_message("ChatBot").markdown(answer)
                st.session_state['messages'].append({"role": "ChatBot", "content": answer})

elif choice == "MedicalBot":
    st.title("Medical AI ChatDoctor")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Medical Model Info")
        st.write("This section uses the 'Gemma-2-2b-it-ChatDoctor' model.")

    with col2:
        st.header("Load Medical Model")
        if st.button("Load Model"):
            st.session_state['medical_model'] = load_medical_model()

    with col3:
        st.header("Model Interaction")
        if st.session_state['medical_model'] is None:
            st.info("Please load the medical model first.")
        else:
            user_input = st.text_input("Ask a medical question:")
            if user_input:
                with st.spinner("Generating response..."):
                    try:
                        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
                        inputs = tokenizer(user_input, return_tensors="pt")
                        response_ids = st.session_state['medical_model'].generate(
                            **inputs, max_length=100, num_return_sequences=1, temperature=0.7
                        )
                        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
                        st.success("Response:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"An error occurred while generating the response: {e}")

elif choice == "TTS":
    st.title("Text-to-Speech Generator")
    tts_input = st.text_area("Enter text to synthesize:")
    voice = 'v2/ru_speaker_5'
    if st.button("Generate Audio"):
        if tts_input.strip():
            with st.spinner("Generating audio..."):
                try:
                    st.session_state['tts_generator'].audio_synthesis(tts_input, voice, output_file="output.wav")
                    st.audio("output.wav", format="audio/wav")
                except Exception as e:
                    st.error(f"An error occurred during audio generation: {e}")
        else:
            st.warning("Please enter some text to synthesize.")

elif choice == "Voice Cloning":
    st.title("Voice Cloning")

    uploaded_voice = st.file_uploader("Upload a voice sample (WAV format)", type=["wav"])
    if uploaded_voice is not None:
        st.success("Voice sample uploaded successfully!")
        speaker_id = st.text_input("Enter a name for the speaker:", value="new_speaker")
        st.session_state['uploaded_voice'] = uploaded_voice
    else:
        st.warning("Please upload a voice sample.")
        speaker_id = "default_speaker"


    input_text = st.text_area("Enter text for voice cloning:", height=100)
    output_file = "cloned_audio.wav"

    if st.button("Clone Voice"):
        if input_text.strip() == "":
            st.warning("Please enter some text for voice cloning.")
        elif uploaded_voice is None:
            st.warning("Please upload a voice sample.")
        else:
            with st.spinner("Cloning voice..."):
                try:
                    voice_cloner = VoiceCloner()
                    speaker_embedding = voice_cloner.process_voice_sample(uploaded_voice)
                    voice_cloner.add_speaker_embedding(speaker_id, speaker_embedding)

                    output_file, sample_rate = voice_cloner.clone_voice(
                        text=input_text, speaker_id=speaker_id, output_file=output_file
                    )

                    st.success("Voice cloned successfully!")
                    audio_file = open(output_file, "rb")
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/wav", start_time=0)
                    st.download_button(
                        label="Download Cloned Audio",
                        data=audio_bytes,
                        file_name=output_file,
                        mime="audio/wav"
                    )
                except Exception as e:
                    st.error(f"An error occurred during voice cloning: {e}")
