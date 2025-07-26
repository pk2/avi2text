# -*- coding: utf-8 -*-

import os
import whisper
import moviepy.editor as mp
import argparse
import sys
import tempfile

def transkrybuj_wideo(sciezka_pliku_wideo: str, model_whisper: str = "base"):
    """
    Wyodrębnia dźwięk z pliku wideo, dokonuje transkrypcji mowy na tekst w języku polskim
    i drukuje wynik na konsoli.

    Wymagania:
    - ffmpeg: Upewnij się, że jest zainstalowany w Twoim systemie i dostępny w PATH.
    - Biblioteki Python: openai-whisper, moviepy

    Args:
        sciezka_pliku_wideo (str): Pełna ścieżka do pliku wideo (np. .avi, .mp4).
        model_whisper (str): Nazwa modelu Whisper do użycia (np. "tiny", "base", "small", "medium", "large").
                             Większe modele są dokładniejsze, ale wolniejsze i wymagają więcej zasobów.
    """
    print(f"--- Rozpoczynanie transkrypcji pliku: {sciezka_pliku_wideo} ---")

    # --- Krok 1: Sprawdzenie, czy plik wideo istnieje ---
    if not os.path.exists(sciezka_pliku_wideo):
        print(f"BŁĄD: Plik '{sciezka_pliku_wideo}' nie został znaleziony.")
        sys.exit(1)

    # --- Krok 2: Wyodrębnienie ścieżki audio z pliku wideo ---
    print("Krok 1/3: Wyodrębnianie ścieżki audio z pliku wideo...")
    try:
        wideo = mp.VideoFileClip(sciezka_pliku_wideo)
        if wideo.audio is None:
            print(f"BŁĄD: Plik wideo '{sciezka_pliku_wideo}' nie zawiera ścieżki audio.")
            sys.exit(1)
            
        # Używamy pliku tymczasowego do przechowywania wyodrębnionego audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio_file:
            sciezka_pliku_audio = tmp_audio_file.name
            wideo.audio.write_audiofile(sciezka_pliku_audio, logger=None)
        
        print(f"Audio zostało pomyślnie zapisane w pliku tymczasowym: {sciezka_pliku_audio}")
        
    except Exception as e:
        print(f"BŁĄD: Wystąpił problem podczas przetwarzania wideo z użyciem moviepy: {e}")
        print("Upewnij się, że masz zainstalowany program 'ffmpeg' w swoim systemie.")
        sys.exit(1)

    # --- Krok 3: Transkrypcja audio przy użyciu Whisper ---
    try:
        print(f"Krok 2/3: Ładowanie modelu Whisper ('{model_whisper}')...")
        # Przy pierwszym uruchomieniu model zostanie pobrany. Może to chwilę potrwać.
        model = whisper.load_model(model_whisper)
        print("Model załadowany. Rozpoczynanie transkrypcji (to może potrwać)...")

        # Dokonaj transkrypcji, jawnie wskazując język polski
        wynik = model.transcribe(sciezka_pliku_audio, language="polish", fp16=False)
        
        transkrypcja = wynik["text"]

        print("\n--- Krok 3/3: Zakończono transkrypcję! ---")
        print("\n--- OTRZYMANY TEKST: ---\n")
        print(transkrypcja)

    except Exception as e:
        print(f"BŁĄD: Wystąpił problem podczas transkrypcji z użyciem Whisper: {e}")
    finally:
        # --- Krok 4: Sprzątanie - usunięcie tymczasowego pliku audio ---
        if 'sciezka_pliku_audio' in locals() and os.path.exists(sciezka_pliku_audio):
            os.remove(sciezka_pliku_audio)
            print(f"\nPlik tymczasowy '{sciezka_pliku_audio}' został usunięty.")


if __name__ == "__main__":
    # Konfiguracja parsera argumentów linii poleceń
    parser = argparse.ArgumentParser(
        description="Transkrypcja mowy z pliku wideo (np. AVI) na tekst w języku polskim.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "sciezka_wideo",
        type=str,
        help="Ścieżka do pliku wideo, który chcesz przetworzyć."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Wybierz model Whisper do użycia.\n"
             "Dostępne opcje (od najszybszego do najdokładniejszego):\n"
             "- tiny: Najszybszy, najniższa dokładność.\n"
             "- base: Dobry kompromis między szybkością a dokładnością (domyślny).\n"
             "- small: Wolniejszy, ale bardziej dokładny.\n"
             "- medium: Znacznie dokładniejszy, ale wolniejszy i wymaga więcej VRAM/RAM.\n"
             "- large: Najlepsza dokładność, najbardziej zasobożerny."
    )

    args = parser.parse_args()

    # Wywołanie głównej funkcji
    transkrybuj_wideo(args.sciezka_wideo, args.model)

