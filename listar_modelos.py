"""
Listar modelos disponíveis na API do Google Gemini
"""
import google.generativeai as genai
import os

# Configurar API key
chave_api = os.getenv("GEMINI_API_KEY") or "AIzaSyB2vrJc5nZLRO_Tx3bPkUr-6DEB8RNGTSc"
genai.configure(api_key=chave_api)

print("\n" + "="*80)
print("MODELOS DISPONÍVEIS NO GOOGLE GEMINI")
print("="*80 + "\n")

try:
    # Listar todos os modelos
    print("A listar modelos disponíveis...\n")

    for modelo in genai.list_models():
        if 'generateContent' in modelo.supported_generation_methods:
            print(f"✓ {modelo.name}")
            print(f"  Descrição: {modelo.display_name}")
            print(f"  Métodos: {', '.join(modelo.supported_generation_methods)}")
            print()

except Exception as e:
    print(f"✗ Erro ao listar modelos: {e}")
    print("\nPossíveis causas:")
    print("  1. Chave API inválida ou expirada")
    print("  2. API Gemini não está ativa para esta chave")
    print("  3. Restrições de região ou quota")
    print()
    print(f"Chave usada: {chave_api[:20]}...")

print("="*80 + "\n")

