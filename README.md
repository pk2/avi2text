# Skrypt do Transkrypcji Wideo z Rozpoznawaniem Mówców

Ten zaawansowany skrypt w języku Python służy do automatycznej transkrypcji plików wideo (np. `.avi`, `.mp4`). Jego główne funkcje to:

-   **Ekstrakcja audio**: Automatycznie wyodrębnia ścieżkę dźwiękową z pliku wideo.
-   **Transkrypcja mowy na tekst**: Wykorzystuje model `WhisperX` do precyzyjnej zamiany mowy na tekst.
-   **Rozpoznawanie mówców (Diarization)**: Identyfikuje, kto mówi w danym momencie i przypisuje mu odpowiednie kwestie.
-   **Zapis postępu**: Tworzy folder roboczy (`nazwa_pliku_work`), w którym zapisuje wyniki pośrednie. W razie błędu lub przerwania pracy, skrypt można uruchomić ponownie, a on wznowi działanie od ostatniego udanego etapu.
-   **Zapis do `.docx`**: Generuje czytelny plik Word z finalną transkrypcją, z pogrubionymi etykietami mówców.
-   **Konfiguracja przez `.env`**: Umożliwia łatwe zarządzanie ustawieniami za pomocą pliku `.env`.
-   **Zaawansowane optymalizacje**: Obsługuje VAD (Wykrywanie Aktywności Głosowej) oraz silnik CTranslate2 dla maksymalnej wydajności na CPU.

## Wymagania

1.  **Python**: Wersja 3.8 lub nowsza.
2.  **FFMPEG**: Niezbędny do przetwarzania plików wideo. Musi być zainstalowany w systemie i dostępny w zmiennej `PATH`.
3.  **Karta graficzna NVIDIA (Opcjonalnie, ale zalecane)**: Znacząco przyspiesza proces.

## Instalacja

1.  **Sklonuj lub pobierz repozytorium** i przejdź do folderu ze skryptem.

2.  **Zainstaluj PyTorch**: Wejdź na [oficjalną stronę PyTorch](https://pytorch.org/get-started/locally/) i wybierz konfigurację odpowiednią dla Twojego systemu (np. `Stable`, `Pip`, `Python`, `CUDA` lub `CPU`). Skopiuj i uruchom wygenerowane polecenie.

3.  **Zainstaluj pozostałe zależności** za pomocą pliku `requirements.txt`:
    ```bash
    git clone https://github.com/pk2/avi2text.git
    cd avi2text
    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    ```

## Konfiguracja

Przed pierwszym uruchomieniem należy skonfigurować plik `.env`.

1.  **Utwórz plik `.env`** w tym samym folderze, w którym znajduje się skrypt.

2.  **Skopiuj i wklej** poniższą zawartość do pliku `.env` i **uzupełnij swój token**:
    ```
    # Plik konfiguracyjny dla skryptu transkrypcji
    HUGGING_FACE_TOKEN="hf_TWOJ_SKOPIOWANY_TOKEN"
    DEFAULT_MODEL="large-v2"
    DEFAULT_SPEAKERS=2
    DEFAULT_LANGUAGE="pl"
    ```

3.  **Uzupełnij `HUGGING_FACE_TOKEN`**:
    -   Zaloguj się lub zarejestruj na [Hugging Face](https://huggingface.co/).
    -   Zaakceptuj regulaminy na stronach modeli [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) oraz [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
    -   Skopiuj swój token z **Settings -> Access Tokens** i wklej go do pliku `.env`.

## Użycie

### Podstawowe użycie
Skrypt użyje ustawień z pliku `.env` oraz automatycznie wykryje najlepsze opcje dla Twojego sprzętu.
```bash
python3 avi2text.py "sciezka/do/mojego_wideo.mp4"

Zaawansowane opcje i optymalizacja
Możesz dostosować działanie skryptu za pomocą flag:

--liczba_mowcow LICZBA: Określa dokładną liczbę mówców.

--model MODEL: Wybiera inny model Whisper (np. medium, small).

--batch_size ROZMIAR: (Tylko GPU) Ustawia liczbę segmentów przetwarzanych naraz. Zwiększ (np. do 16, 32), jeśli masz GPU z dużą ilością VRAM.

--cpu_threads LICZBA: (Tylko CPU) Ustawia liczbę wątków procesora. Domyślnie używa wszystkich dostępnych.

--no-vad: Wyłącza filtr VAD (domyślnie włączony).

--compute_type TYP: Zmienia typ obliczeń. Aby użyć CTranslate2 na CPU i uzyskać 4x przyspieszenie, użyj --compute_type int8.

Przykład maksymalnej optymalizacji na CPU:

python3 avi2text.py "dlugie_nagranie.avi" --model medium --compute_type int8

