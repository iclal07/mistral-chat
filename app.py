import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Model ve tokenizer'ı yükleyin, token ile kimlik doğrulaması yapın
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", use_auth_token=token)

# Kullanıcıdan gelen soruya model ile yanıt üretelim
def generate_response(question):
    inputs = tokenizer.encode(question, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, do_sample=True, top_p=0.95, top_k=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit arayüzü
def main():
    st.title("Mixtral Chatbot")
    
    st.write("Bu chatbot, Mixtral modelini kullanarak doğal dilde sorularınızı yanıtlar.")
    
    # Kullanıcıdan bir soru al
    user_input = st.text_input("Sorunuzu yazın:")
    
    if user_input:
        with st.spinner("Yanıt üretiliyor..."):
            response = generate_response(user_input)
            st.write("Cevap:", response)

if __name__ == "__main__":
    main()
