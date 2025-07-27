# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import argparse
import moviepy.editor as mp
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from docx import Document
import json
import pickle
from dotenv import load_dotenv
import logging

def transkrybuj_i_rozpoznaj_mowcow(
    sciezka_pliku_wideo: str,
    sciezka_pliku_docx: str,
    liczba_mowcow: int,
    model_whisper: str,
    jezyk: str,
    batch_size: int,
    compute_type: str,
    asr_options: dict
):
    """
    Wyodrębnia dźwięk z pliku wideo, dokonuje transkrypcji, dzieli tekst na mówców
    i zapisuje wynik do pliku .docx. Zapisuje postęp, aby móc wznowić pracę.
    """
    # --- Konfiguracja logowania w celu wyłączenia niepotrzebnych ostrzeżeń ---
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    logging.getLogger('pyannote').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)


    print(f"--- Rozpoczynanie transkrypcji pliku: {sciezka_pliku_wideo} ---")

    # --- Krok 1: Konfiguracja folderu roboczego do zapisywania postępu ---
    nazwa_pliku_bazowa = os.path.splitext(os.path.basename(sciezka_pliku_wideo))[0]
    folder_roboczy = f"{nazwa_pliku_bazowa}_work"
    os.makedirs(folder_roboczy, exist_ok=True)
    print(f"Używam folderu roboczego: {folder_roboczy}")

    # Definicja ścieżek do plików pośrednich
    sciezka_pliku_audio = os.path.join(folder_roboczy, "audio.flac")
    sciezka_transkrypcji = os.path.join(folder_roboczy, "transkrypcja.json")
    sciezka_aligned = os.path.join(folder_roboczy, "aligned.json")
    sciezka_diarization = os.path.join(folder_roboczy, "diarization.pkl")
    sciezka_wyniku_finalnego = os.path.join(folder_roboczy, "wynik_finalny.json")

    # --- Krok 2: Wyodrębnienie ścieżki audio do formatu FLAC (z wznawianiem) ---
    if not os.path.exists(sciezka_pliku_audio):
        print("Krok 1/7: Wyodrębnianie ścieżki audio...")
        if not os.path.exists(sciezka_pliku_wideo):
            print(f"BŁĄD: Plik '{sciezka_pliku_wideo}' nie został znaleziony.")
            sys.exit(1)
        try:
            wideo = mp.VideoFileClip(sciezka_pliku_wideo)
            if wideo.audio is None:
                print(f"BŁĄD: Plik wideo '{sciezka_pliku_wideo}' nie zawiera ścieżki audio.")
                sys.exit(1)
            wideo.audio.write_audiofile(sciezka_pliku_audio, codec='flac', logger=None)
            print(f"Audio zostało pomyślnie zapisane w formacie FLAC: {sciezka_pliku_audio}")
        except Exception as e:
            print(f"BŁĄD podczas przetwarzania wideo: {e}")
            sys.exit(1)
    else:
        print("Krok 1/7: Pomijanie ekstrakcji audio (plik już istnieje).")

    # --- Krok 3: Konfiguracja urządzenia (GPU lub CPU) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Krok 2/7: Używane urządzenie: {device}")
    if device == "cpu":
        print("OSTRZEŻENIE: Brak GPU. Przetwarzanie na CPU będzie znacznie wolniejsze. Rozważ użycie CTranslate2 (`--compute_type int8`).")
    
    # --- Krok 4: Transkrypcja (z wznawianiem) ---
    if not os.path.exists(sciezka_transkrypcji):
        print(f"Krok 3/7: Ładowanie modelu WhisperX ('{model_whisper}')...")
        model = None
        try:
            model = whisperx.load_model(model_whisper, device, compute_type=compute_type, asr_options=asr_options)
            audio = whisperx.load_audio(sciezka_pliku_audio)
            
            print("Krok 4/7: Rozpoczynanie transkrypcji (to może potrwać)...")
            wynik_transkrypcji = model.transcribe(audio, batch_size=batch_size, language=jezyk, print_progress=True)
            
            with open(sciezka_transkrypcji, 'w', encoding='utf-8') as f:
                json.dump(wynik_transkrypcji, f, ensure_ascii=False, indent=4)
            print(f"Transkrypcja zakończona i zapisana w: {sciezka_transkrypcji}")
        except Exception as e:
            print(f"BŁĄD podczas transkrypcji: {e}")
            sys.exit(1)
        finally:
            if model is not None: del model
    else:
        print("Kroki 3/7 i 4/7: Pomijanie transkrypcji (pliki już istnieją).")
        with open(sciezka_transkrypcji, 'r', encoding='utf-8') as f:
            wynik_transkrypcji = json.load(f)

    # --- Krok 5: Podział na mówców (z wznawianiem) ---
    if not os.path.exists(sciezka_wyniku_finalnego):
        model_a = None
        diarize_model = None
        try:
            # Wyrównywanie
            if not os.path.exists(sciezka_aligned):
                print("Krok 5/7: Wyrównywanie transkrypcji i przywracanie interpunkcji...")
                model_a, metadata = whisperx.load_align_model(language_code=wynik_transkrypcji["language"], device=device)
                wynik_aligned = whisperx.align(wynik_transkrypcji["segments"], model_a, metadata, whisperx.load_audio(sciezka_pliku_audio), device, return_char_alignments=False)
                with open(sciezka_aligned, 'w', encoding='utf-8') as f:
                    json.dump(wynik_aligned, f, ensure_ascii=False, indent=4)
                print(f"Wyrównywanie zakończone i zapisane w: {sciezka_aligned}")
                del model_a
            else:
                print("Krok 5/7: Pomijanie wyrównywania (plik już istnieje).")
                with open(sciezka_aligned, 'r', encoding='utf-8') as f:
                    wynik_aligned = json.load(f)

            # Diarization
            if not os.path.exists(sciezka_diarization):
                print("Krok 6/7: Rozpoznawanie mówców (diarization)...")
                hf_token = os.environ.get("HUGGING_FACE_TOKEN")
                if hf_token is None:
                    print("\nBŁĄD KRYTYCZNY: Brak tokena HUGGING_FACE_TOKEN w pliku .env lub zmiennych środowiskowych.")
                    sys.exit(1)
                diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
                diarize_segments = diarize_model(sciezka_pliku_audio, min_speakers=liczba_mowcow, max_speakers=liczba_mowcow)
                with open(sciezka_diarization, 'wb') as f:
                    pickle.dump(diarize_segments, f)
                print(f"Diarization zakończony i zapisany w: {sciezka_diarization}")
                del diarize_model
            else:
                print("Krok 6/7: Pomijanie diarization (plik już istnieje).")
                with open(sciezka_diarization, 'rb') as f:
                    diarize_segments = pickle.load(f)

            # Przypisywanie mówców
            wynik_finalny = whisperx.assign_word_speakers(diarize_segments, wynik_aligned)
            with open(sciezka_wyniku_finalnego, 'w', encoding='utf-8') as f:
                json.dump(wynik_finalny, f, ensure_ascii=False, indent=4)
            print(f"Przypisano mówców i zapisano wynik w: {sciezka_wyniku_finalnego}")

        except Exception as e:
            print(f"\nBŁĄD podczas rozpoznawania mówców: {e}")
            sys.exit(1)
    else:
        print("Kroki 5/7 i 6/7: Pomijanie (finalny plik z mówcami już istnieje).")
        with open(sciezka_wyniku_finalnego, 'r', encoding='utf-8') as f:
            wynik_finalny = json.load(f)

    # --- Krok 7: Zapis do pliku Word ---
    print(f"Krok 7/7: Zapisywanie wyniku do pliku {sciezka_pliku_docx}...")
    try:
        doc = Document()
        doc.add_heading(f'Transkrypcja pliku: {os.path.basename(sciezka_pliku_wideo)}', 0)
        
        aktualny_mowca = None
        aktualna_kwestia = []

        # ULEPSZONA LOGIKA: Składanie tekstu słowo po słowie dla lepszej interpunkcji
        for segment in wynik_finalny["segments"]:
            if "speaker" not in segment:
                segment['speaker'] = "MÓWCA_NIEZNANY"
            
            segment_speaker = segment["speaker"].replace("SPEAKER_", "Mówca ")

            if aktualny_mowca != segment_speaker and aktualny_mowca is not None:
                p = doc.add_paragraph()
                p.add_run(f"{aktualny_mowca}:").bold = True
                p.add_run(f" {''.join(aktualna_kwestia).lstrip()}")
                aktualna_kwestia = []
            
            aktualny_mowca = segment_speaker
            
            # Składamy zdanie ze słów w segmencie
            if 'words' in segment:
                for word_info in segment['words']:
                    # Dodajemy słowo i spację po nim
                    aktualna_kwestia.append(word_info['word'] + " ")
            else:
                # Fallback, jeśli nie ma informacji o słowach
                aktualna_kwestia.append(segment.get('text', ' (słowo niezrozumiałe) ').strip() + " ")

        if aktualny_mowca is not None and aktualna_kwestia:
            p = doc.add_paragraph()
            p.add_run(f"{aktualny_mowca}:").bold = True
            p.add_run(f" {''.join(aktualna_kwestia).lstrip()}")

        doc.save(sciezka_pliku_docx)
        print("\n--- Zakończono pomyślnie! ---")
        print(f"Wynik został zapisany w pliku: {sciezka_pliku_docx}")

    except Exception as e:
        print(f"BŁĄD podczas zapisu do pliku .docx: {e}")

if __name__ == "__main__":
    # Wczytaj zmienne z pliku .env
    load_dotenv()

    # Domyślny typ obliczeń w zależności od dostępności GPU
    default_compute_type = "float16" if torch.cuda.is_available() else "int8"

    parser = argparse.ArgumentParser(
        description="Zaawansowana transkrypcja wideo z podziałem na mówców i zapisem do Word.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("sciezka_wideo", type=str, help="Ścieżka do pliku wideo.")
    parser.add_argument(
        "--liczba_mowcow", 
        type=int, 
        default=os.getenv("DEFAULT_SPEAKERS", None),
        help="Liczba mówców. Nadpisuje wartość z pliku .env."
    )
    parser.add_argument(
        "--plik_wyjsciowy",
        type=str,
        default=None,
        help="Nazwa pliku .docx. Domyślnie [nazwa_wideo].docx"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("DEFAULT_MODEL", "large-v2"),
        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
        help="Model Whisper do użycia. Nadpisuje wartość z pliku .env."
    )
    parser.add_argument(
        "--jezyk",
        type=str,
        default=os.getenv("DEFAULT_LANGUAGE", "pl"),
        help="Kod języka (np. 'pl', 'en'). Nadpisuje wartość z pliku .env."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Liczba segmentów przetwarzanych równolegle. Zwiększ dla lepszej wydajności na GPU."
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=os.cpu_count(),
        help="Liczba wątków CPU do użycia podczas transkrypcji (tylko dla trybu CPU)."
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default=default_compute_type,
        choices=["float16", "float32", "int8", "int8_float16"],
        help=f"Typ obliczeń. Domyślnie: '{default_compute_type}'. Użyj 'int8' dla CTranslate2 na CPU."
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Liczba 'promieni' w algorytmie beam search. Zwiększenie poprawia dokładność kosztem prędkości."
    )
    args = parser.parse_args()

    # Sprawdzenie, czy liczba mówców została podana
    if args.liczba_mowcow is None:
        print("BŁĄD: Liczba mówców jest wymagana. Ustaw ją w pliku .env jako DEFAULT_SPEAKERS lub podaj za pomocą flagi --liczba_mowcow.")
        sys.exit(1)
        
    # Ustaw domyślną nazwę pliku wyjściowego, jeśli nie została podana
    plik_wyjsciowy = args.plik_wyjsciowy
    if plik_wyjsciowy is None:
        nazwa_bazowa = os.path.splitext(os.path.basename(args.sciezka_wideo))[0]
        plik_wyjsciowy = f"{nazwa_bazowa}.docx"
        
    # Ustawienie liczby wątków dla PyTorch, jeśli pracujemy na CPU
    if not torch.cuda.is_available():
        print(f"Ustawiam liczbę wątków CPU na: {args.cpu_threads}")
        torch.set_num_threads(args.cpu_threads)

    # Przygotowanie opcji dla modelu ASR (Automatic Speech Recognition)
    asr_options = {
        "beam_size": args.beam_size,
    }

    transkrybuj_i_rozpoznaj_mowcow(
        args.sciezka_wideo,
        plik_wyjsciowy,
        int(args.liczba_mowcow),
        args.model,
        args.jezyk,
        args.batch_size,
        args.compute_type,
        asr_options
    )

