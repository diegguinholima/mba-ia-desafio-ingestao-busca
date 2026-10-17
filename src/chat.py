from search import search_prompt

def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (digite 'sair' para encerrar):")

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("sair", "exit", "quit"):
            break

        try:
            print(f"RESPOSTA: {chain.invoke(question)}")
        except Exception as e:
            print(f"Erro ao processar a pergunta: {e}")

if __name__ == "__main__":
    main()
