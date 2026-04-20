#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_apis.py
============
Verifica que todas las APIs configuradas responden correctamente.
Hace UNA llamada minima a cada proveedor/modelo y muestra el resultado.

Uso:
    python test_apis.py
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# ── Colores ANSI (compatibles con Windows si se usa Windows Terminal) ─────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

OK   = f"{GREEN}[OK]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
SKIP = f"{YELLOW}[SKIP]{RESET}"

MENSAJE_TEST = "Responde solo con la palabra: FUNCIONA"

results = []


def _print_result(label: str, status: str, info: str = "") -> None:
    line = f"  {status}  {label:<35}"
    if info:
        # Truncar a 80 chars para no romper la consola
        info_clean = info.replace("\n", " ").strip()[:80]
        line += f"  -> {info_clean}"
    print(line)
    results.append((label, status, info))


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────────────────
def _test_openai_model(client, model_id: str, supports_temp: bool, supports_max_tokens: bool) -> tuple:
    kwargs = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Eres un asistente de prueba."},
            {"role": "user",   "content": MENSAJE_TEST},
        ],
    }
    if supports_temp:
        kwargs["temperature"] = 0.0
    if supports_max_tokens:
        kwargs["max_tokens"] = 20
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


def test_openai() -> None:
    print(f"\n{BOLD}OpenAI{RESET}")
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        _print_result("OpenAI (todos)", SKIP, "OPENAI_API_KEY no encontrada en .env")
        return
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
    except ImportError:
        _print_result("OpenAI (todos)", SKIP, "openai no instalado")
        return

    models = [
        ("gpt-4o",              True,  True),
        ("gpt-4.1",             True,  True),
        ("gpt-5.3-chat-latest", False, False),
    ]
    for model_id, supports_temp, supports_max_tokens in models:
        try:
            answer = _test_openai_model(client, model_id, supports_temp, supports_max_tokens)
            _print_result(f"OpenAI / {model_id}", OK, answer)
        except Exception as e:
            _print_result(f"OpenAI / {model_id}", FAIL, str(e))
        time.sleep(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic
# ─────────────────────────────────────────────────────────────────────────────
def test_anthropic() -> None:
    print(f"\n{BOLD}Anthropic{RESET}")
    key = os.getenv("CLAUDE_API_KEY", "")
    if not key:
        _print_result("Anthropic (todos)", SKIP, "CLAUDE_API_KEY no encontrada en .env")
        return
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
    except ImportError:
        _print_result("Anthropic (todos)", SKIP, "anthropic no instalado (pip install anthropic)")
        return

    models = [
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
    ]
    for model_id in models:
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=20,
                temperature=0.0,
                system="Eres un asistente de prueba.",
                messages=[{"role": "user", "content": MENSAJE_TEST}],
            )
            answer = resp.content[0].text.strip()
            _print_result(f"Anthropic / {model_id}", OK, answer)
        except Exception as e:
            _print_result(f"Anthropic / {model_id}", FAIL, str(e))
        time.sleep(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Google Generative AI
# ─────────────────────────────────────────────────────────────────────────────
def test_google() -> None:
    print(f"\n{BOLD}Google Generative AI{RESET}")
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        _print_result("Google (todos)", SKIP, "GEMINI_API_KEY no encontrada en .env")
        return
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
    except ImportError:
        _print_result("Google (todos)", SKIP, "google-generativeai no instalado")
        return

    # gemini-2.5-pro es un modelo de "thinking" que necesita mas tokens
    models = [
        ("gemini-2.5-flash", 500),
        ("gemini-2.5-pro",   4096),
    ]
    for model_id, max_tok in models:
        try:
            gen_config = genai.GenerationConfig(max_output_tokens=max_tok, temperature=0.0)
            model = genai.GenerativeModel(
                model_name=model_id,
                system_instruction="Eres un asistente de prueba.",
                generation_config=gen_config,
            )
            resp = model.generate_content(MENSAJE_TEST)
            # Acceso robusto: evita crash si la respuesta esta bloqueada
            if resp.candidates and resp.candidates[0].content.parts:
                answer = resp.candidates[0].content.parts[0].text.strip()
            else:
                finish = resp.candidates[0].finish_reason if resp.candidates else "unknown"
                answer = f"[sin contenido, finish_reason={finish}]"
            _print_result(f"Google / {model_id}", OK, answer)
        except Exception as e:
            _print_result(f"Google / {model_id}", FAIL, str(e))
        time.sleep(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────
def test_deepseek() -> None:
    print(f"\n{BOLD}DeepSeek{RESET}")
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        _print_result("DeepSeek / deepseek-chat", SKIP, "DEEPSEEK_API_KEY no encontrada en .env")
        return
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
    except ImportError:
        _print_result("DeepSeek / deepseek-chat", SKIP, "openai no instalado")
        return

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un asistente de prueba."},
                {"role": "user",   "content": MENSAJE_TEST},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        answer = resp.choices[0].message.content.strip()
        _print_result("DeepSeek / deepseek-chat", OK, answer)
    except Exception as e:
        _print_result("DeepSeek / deepseek-chat", FAIL, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print()
    print("=" * 62)
    print("  Test de conectividad de APIs")
    print("=" * 62)

    test_openai()
    test_anthropic()
    test_google()
    test_deepseek()

    # Resumen final
    ok_count   = sum(1 for _, s, _ in results if s == OK)
    fail_count = sum(1 for _, s, _ in results if s == FAIL)
    skip_count = sum(1 for _, s, _ in results if s == SKIP)

    print()
    print("=" * 62)
    print(f"  Resultado: {ok_count} OK  |  {fail_count} FAIL  |  {skip_count} SKIP")
    print("=" * 62)
    print()

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
