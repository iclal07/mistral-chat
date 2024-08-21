from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Mistral 7B modelini ve tokenizer'ı yükleme
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def mistral_chatbot(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150, do_sample=True, top_k=50, top_p=0.95, temperature=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    print("Mistral 7B Chatbot'a hoş geldiniz! Çıkmak için 'exit' yazın.")
    while True:
        user_input = input("Siz: ")
        if user_input.lower() == 'exit':
            print("Chatbot kapatılıyor...")
            break
        response = mistral_chatbot(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()