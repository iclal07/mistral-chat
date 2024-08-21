from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model ve tokenizer'ı yükleyin
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

# Yanıt oluşturma fonksiyonu
def generate_response(question, max_length=200):
    inputs = tokenizer.encode(f"User: {question}\nBot:", return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_p=0.95, top_k=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Cevap "Bot:" kelimesinden başlasın
    response = response.split("Bot:")[-1].strip()
    return response

# Chatbot döngüsü
def chat():
    print("Chatbot'a hoş geldiniz! Çıkış yapmak için 'exit' yazın.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Chatbot'tan çıkılıyor... Görüşmek üzere!")
            break
        
        response = generate_response(user_input)
        print(f"Bot: {response}")

# Chatbot'u başlat
if __name__ == "__main__":
    chat()
