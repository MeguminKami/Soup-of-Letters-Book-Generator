"""
Módulo de Geração de Palavras IA para Gerador de Sopa de Letras

Este módulo fornece geração de palavras com IA incluindo:
- Geração de palavras baseada em temas
- Prevenção de duplicados (incluindo plurais)
- Registo de mensagens para depuração
- Consistência multi-página
"""

import os
import json
import re
import logging
from typing import Dict, Any, List, Set, Tuple, Counter as CounterType
from collections import Counter
from datetime import datetime
try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("Por favor instale o Google Generative AI: pip install google-generativeai")

# Configurar registo
DIRETORIO_LOGS = "ai_logs"
os.makedirs(DIRETORIO_LOGS, exist_ok=True)

# Criar ficheiro de log com timestamp
nome_ficheiro_log = os.path.join(DIRETORIO_LOGS, f"geracao_ia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(nome_ficheiro_log, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GeradorPalavrasIA:
    """
    Gerador de palavras com IA com suporte a temas e prevenção de duplicados.
    """

    def __init__(self, chave_api: str = None, modelo: str = None):
        """
        Inicializar o gerador de palavras IA.

        Args:
            chave_api: Chave API do Google Gemini (padrão: variável de ambiente GEMINI_API_KEY ou chave incorporada)
            modelo: Nome do modelo (padrão: gemini-1.5-flash)
        """
        # Determinar chave API
        if chave_api is None:
            # Usar chave fornecida ou variável de ambiente
            chave_api = os.getenv("GEMINI_API_KEY") or "x"

        if not chave_api:
            raise RuntimeError(
                "Modo IA requer uma chave API do Google Gemini. Defina a variável de ambiente GEMINI_API_KEY."
            )

        # Configurar cliente Gemini
        genai.configure(api_key=chave_api)
        # Usar modelo disponível da lista (gemini-2.0-flash é rápido e gratuito)
        nome_modelo = modelo or os.getenv("AI_MODEL", "models/gemini-2.0-flash")
        self.modelo = nome_modelo
        self.cliente = genai.GenerativeModel(nome_modelo)
        self.id_sessao = datetime.now().strftime('%Y%m%d_%H%M%S')

        logger.info(f"Gerador de Palavras IA inicializado com modelo Gemini: {self.modelo}")
        logger.info(f"ID da Sessão: {self.id_sessao}")
        logger.info(f"Ficheiro de log: {nome_ficheiro_log}")

    def gerar_palavras(
        self,
        quantidade: int,
        plano_comprimentos: List[int],
        temas: List[str],
        palavras_usadas: Set[str],
        numero_pagina: int = 1,
        max_tentativas: int = 6
    ) -> List[str]:
        """
        Gerar palavras usando IA com suporte a temas e prevenção de duplicados.

        Args:
            quantidade: Número total de palavras necessárias
            plano_comprimentos: Lista de comprimentos de palavras requeridos (ex: [4, 4, 5, 5, 6])
            temas: Lista de temas a seguir (pode estar vazia)
            palavras_usadas: Conjunto de palavras já usadas (incluindo plurais)
            numero_pagina: Número da página atual para registo
            max_tentativas: Número máximo de tentativas

        Returns:
            Lista de palavras geradas
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"A GERAR PALAVRAS - Página {numero_pagina}")
        logger.info(f"{'='*80}")
        logger.info(f"Necessário: {quantidade} palavras")
        logger.info(f"Distribuição de comprimentos: {Counter(plano_comprimentos)}")
        logger.info(f"Temas: {temas if temas else 'Nenhum (aleatório)'}")
        logger.info(f"Palavras já usadas: {len(palavras_usadas)}")

        # Construir descrição dos requisitos de comprimento
        contagem_comprimentos = Counter(plano_comprimentos)
        desc_comprimentos = ", ".join(
            f"{qtd} palavra{'s' if qtd > 1 else ''} com {comp} letras"
            for comp, qtd in sorted(contagem_comprimentos.items())
        )

        # Construir descrição de tema
        if temas:
            desc_tema = f"com os temas: {', '.join(temas)}"
        else:
            desc_tema = "sem tema específico (palavras aleatórias)"

        # Construir lista de palavras bloqueadas (palavras recentes para contexto)
        lista_bloqueadas = list(palavras_usadas)[-100:] if palavras_usadas else []
        desc_bloqueadas = ", ".join(lista_bloqueadas) if lista_bloqueadas else "nenhuma"

        # Mensagem do sistema
        msg_sistema = {
            "role": "system",
            "content": (
                "És um gerador preciso de palavras para puzzles de sopa de letras. "
                "Gera APENAS JSON válido no formato exato solicitado. "
                "Regras:\n"
                "- Todas as palavras devem ser apenas letras ASCII maiúsculas (A-Z)\n"
                "- Sem espaços, hífens, acentos ou caracteres especiais\n"
                "- Usa apenas substantivos comuns (sem nomes próprios)\n"
                "- Evita formas plurais se o singular existir (GATOS vs GATO: escolhe GATO)\n"
                "- Cada palavra deve ser única\n"
                "- As palavras devem corresponder ao(s) tema(s) especificado(s)\n"
                "- Não reutilizes nenhuma palavra anteriormente usada ou suas variantes"
            )
        }

        # Mensagem do utilizador
        msg_utilizador = {
            "role": "user",
            "content": (
                f"Gera exatamente {quantidade} palavras únicas {desc_tema}.\n"
                f"Requisitos de comprimento: {desc_comprimentos}.\n"
                f"Palavras usadas anteriormente a evitar: {desc_bloqueadas}.\n"
                f"\nFormato de saída:\n"
                f'{{"palavras": ["PALAVRA1", "PALAVRA2", "PALAVRA3", ...]}}'
            )
        }

        # Construir prompt completo para Gemini
        prompt_completo = f"""
{msg_sistema['content']}

{msg_utilizador['content']}
"""

        logger.info(f"\n--- Prompt Inicial ---")
        logger.info(f"Prompt: {prompt_completo[:300]}...")

        # Ciclo de tentativas
        for tentativa in range(1, max_tentativas + 1):
            logger.info(f"\n--- Tentativa {tentativa}/{max_tentativas} ---")

            try:
                # Chamar IA Gemini
                resposta = self.cliente.generate_content(
                    prompt_completo,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                    )
                )

                resposta_bruta = resposta.text if resposta.text else ""
                logger.info(f"Resposta da IA: {resposta_bruta[:500]}...")

                # Analisar e validar
                palavras, erro = self._analisar_e_validar(
                    resposta_bruta, contagem_comprimentos, palavras_usadas
                )

                if palavras:
                    logger.info(f"✓ SUCESSO - Geradas {len(palavras)} palavras válidas")
                    logger.info(f"Palavras: {', '.join(palavras)}")
                    return palavras

                # Validação falhou - tentar novamente com feedback
                logger.warning(f"✗ VALIDAÇÃO FALHOU: {erro}")

                # Adicionar feedback ao prompt
                prompt_completo += f"\n\nResposta anterior:\n{resposta_bruta[:4000]}\n\nERRO: {erro}\nPor favor, gera JSON corrigido seguindo o formato e regras exatos. Certifica-te que todas as palavras correspondem aos comprimentos e tema requeridos."

            except Exception as e:
                logger.error(f"✗ EXCEÇÃO: {str(e)}")
                if tentativa == max_tentativas:
                    raise

        msg_erro = f"Falha ao gerar palavras válidas após {max_tentativas} tentativas"
        logger.error(f"\n{'='*80}")
        logger.error(f"GERAÇÃO FALHOU - Página {numero_pagina}")
        logger.error(msg_erro)
        logger.error(f"{'='*80}\n")
        raise RuntimeError(msg_erro)

    def _analisar_e_validar(
        self,
        resposta_bruta: str,
        contagens_requeridas: CounterType[int],
        palavras_usadas: Set[str]
    ) -> Tuple[List[str] | None, str]:
        """
        Analisar resposta da IA e validar palavras.

        Returns:
            (palavras, mensagem_erro) - palavras é None se a validação falhar
        """
        # Extrair JSON
        try:
            conteudo = self._extrair_json(resposta_bruta)
            dados = json.loads(conteudo)
        except json.JSONDecodeError as e:
            return None, f"JSON inválido: {str(e)}"

        # Verificar estrutura
        if not isinstance(dados, dict) or "palavras" not in dados:
            return None, "Falta a chave 'palavras' no JSON"

        palavras = dados["palavras"]
        if not isinstance(palavras, list):
            return None, "'palavras' deve ser um array"

        # Validar cada palavra
        palavras_validadas = []
        vistas_no_lote = set()

        for palavra in palavras:
            if not isinstance(palavra, str):
                return None, f"Palavra deve ser string, obteve {type(palavra)}"

            palavra = palavra.strip().upper()

            # Verificar formato (apenas A-Z)
            if not re.match(r'^[A-Z]+$', palavra):
                return None, f"Palavra '{palavra}' contém caracteres inválidos"

            # Verificar duplicados no lote
            if palavra in vistas_no_lote:
                return None, f"Palavra duplicada no lote: '{palavra}'"

            # Verificar contra palavras usadas (correspondência exata)
            if palavra in palavras_usadas:
                return None, f"Palavra já usada: '{palavra}'"

            # Verificar plurais (verificação simples: termina com S)
            if palavra.endswith('S') and palavra[:-1] in palavras_usadas:
                return None, f"Forma plural de palavra usada: '{palavra}' (base: '{palavra[:-1]}')"

            # Verificar se a forma singular foi usada (palavra sem S)
            singular = palavra[:-1] if palavra.endswith('S') else palavra
            if singular in palavras_usadas and singular != palavra:
                return None, f"Forma singular já usada: '{singular}' (plural: '{palavra}')"

            vistas_no_lote.add(palavra)
            palavras_validadas.append(palavra)

        # Verificar distribuição de comprimentos
        contagens_reais = Counter(len(p) for p in palavras_validadas)
        if contagens_reais != contagens_requeridas:
            return None, f"Incompatibilidade de comprimentos. Requerido: {dict(contagens_requeridas)}, Obtido: {dict(contagens_reais)}"

        # Verificar contagem total
        total_esperado = sum(contagens_requeridas.values())
        if len(palavras_validadas) != total_esperado:
            return None, f"Incompatibilidade de contagem. Requerido: {total_esperado}, Obtido: {len(palavras_validadas)}"

        return palavras_validadas, ""

    def _extrair_json(self, texto: str) -> str:
        """Extrair JSON da resposta da IA, tratando code fences e texto extra."""
        texto = texto.strip()

        # Remover code fences
        if texto.startswith("```"):
            texto = re.sub(r"^```[a-zA-Z]*\s*", "", texto)
            texto = re.sub(r"\s*```$", "", texto)
            texto = texto.strip()

        # Se começa com { ou [, devolver como está
        if texto.startswith("{") or texto.startswith("["):
            return texto

        # Tentar encontrar objeto JSON com "palavras"
        obj_match = re.search(r'\{[^{}]*"palavras"[^{}]*\[[^\]]*\][^{}]*\}', texto, re.DOTALL)
        if obj_match:
            return obj_match.group(0)

        # Tentar pesquisa mais ampla
        obj_match = re.search(r'\{.*\}', texto, re.DOTALL)
        if obj_match:
            return obj_match.group(0)

        return texto


def testar_geracao_ia():
    """Testar o gerador de palavras IA."""
    print("\n" + "="*80)
    print("A Testar Gerador de Palavras IA")
    print("="*80 + "\n")

    try:
        gerador = GeradorPalavrasIA()

        # Teste 1: Gerar 14 palavras para uma página de batalha
        print("Teste 1: Página de batalha (14 palavras)")
        palavras1 = gerador.gerar_palavras(
            quantidade=14,
            plano_comprimentos=[4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6],
            temas=["natureza", "animais"],
            palavras_usadas=set(),
            numero_pagina=1
        )
        print(f"Geradas: {', '.join(palavras1)}\n")

        # Teste 2: Gerar mais palavras com as anteriores bloqueadas
        print("Teste 2: Página de batalha (14 palavras, com palavras anteriores bloqueadas)")
        palavras2 = gerador.gerar_palavras(
            quantidade=14,
            plano_comprimentos=[4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6],
            temas=["natureza", "animais"],
            palavras_usadas=set(palavras1),
            numero_pagina=2
        )
        print(f"Geradas: {', '.join(palavras2)}\n")

        # Teste 3: Sem tema
        print("Teste 3: Palavras aleatórias (sem tema)")
        palavras3 = gerador.gerar_palavras(
            quantidade=14,
            plano_comprimentos=[4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6],
            temas=[],
            palavras_usadas=set(palavras1 + palavras2),
            numero_pagina=3
        )
        print(f"Geradas: {', '.join(palavras3)}\n")

        print("="*80)
        print("✓ Todos os testes passaram!")
        print(f"Verifique o ficheiro de log: {nome_ficheiro_log}")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n✗ Teste falhou: {str(e)}\n")
        raise


if __name__ == "__main__":
    testar_geracao_ia()

