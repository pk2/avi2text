# Skrypt do Transkrypcji Wideo z Rozpoznawaniem Mówców

Ten skrypt w języku Python służy do automatycznej transkrypcji plików wideo (np. `.avi`, `.mp4`). Jego główne funkcje to:

-   **Ekstrakcja audio**: Automatycznie wyodrębnia ścieżkę dźwiękową z pliku wideo.
-   **Transkrypcja mowy na tekst**: Wykorzystuje model `WhisperX` do precyzyjnej zamiany mowy na tekst.
-   **Rozpoznawanie mówców (Diarization)**: Identyfikuje, kto mówi w danym momencie i przypisuje mu odpowiednie kwestie.
-   **Zapis postępu**: Tworzy folder roboczy (`nazwa_pliku_work`), w którym zapisuje wyniki pośrednie. W razie błędu lub przerwania pracy, skrypt można uruchomić ponownie, a on wznowi działanie od ostatniego udanego etapu.
-   **Zapis do `.docx`**: Generuje czytelny plik Word z finalną transkrypcją, z pogrubionymi etykietami mówców.
-   **Konfiguracja przez `.env`**: Umożliwia łatwe zarządzanie ustawieniami za pomocą pliku `.env`.

## Wymagania

1.  **Python**: Wersja 3.8 lub nowsza.
2.  **FFMPEG**: Niezbędny do przetwarzania plików wideo. Musi być zainstalowany w systemie i dostępny w zmiennej `PATH`.
    -   **Windows**: Pobierz z [oficjalnej strony](https://ffmpeg.org/download.html).
    -   **macOS (Homebrew)**: `brew install ffmpeg`
    -   **Linux (Debian/Ubuntu)**: `sudo apt update && sudo apt install ffmpeg`
3.  **Karta graficzna NVIDIA (Opcjonalnie, ale zalecane)**: Znacząco przyspiesza proces. W przypadku jej braku, skrypt użyje procesora (CPU), co będzie znacznie wolniejsze.

## Instalacja

1.  **Sklonuj lub pobierz repozytorium** i przejdź do folderu ze skryptem.

    ```bash
    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    ```

## Konfiguracja

Przed pierwszym uruchomieniem należy skonfigurować plik `.env`.

1.  **Utwórz plik `.env`** w tym samym folderze, w którym znajduje się skrypt.

2.  **Skopiuj i wklej** poniższą zawartość do pliku `.env`:
    ```
    # Plik konfiguracyjny dla skryptu transkrypcji
    HUGGING_FACE_TOKEN="hf_TWOJ_SKOPIOWANY_TOKEN"
    DEFAULT_MODEL="large-v2"
    DEFAULT_SPEAKERS=2
    DEFAULT_LANGUAGE="pl"
    ```

3.  **Uzupełnij `HUGGING_FACE_TOKEN`**:
    -   Zaloguj się lub zarejestruj na [Hugging Face](https://huggingface.co/).
    -   Przejdź na strony modeli [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) oraz [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) i zaakceptuj regulaminy.
    -   Przejdź do **Settings -> Access Tokens** w ustawieniach swojego konta i skopiuj token.
    -   Wklej skopiowany token do pliku `.env`. Token jest wymagany do jednorazowego pobrania modeli `pyannote` na dysk. Całe dalsze przetwarzanie odbywa się w 100% lokalnie.

4.  **Dostosuj pozostałe zmienne** (opcjonalnie):
    -   `DEFAULT_MODEL`: Zmień model Whisper, jeśli potrzebujesz (np. na `medium` dla szybszego działania kosztem mniejszej dokładności).
    -   `DEFAULT_SPEAKERS`: Ustaw domyślną liczbę mówców.
    -   `DEFAULT_LANGUAGE`: Ustaw domyślny język transkrypcji (np. `en` dla angielskiego).

## Użycie

Skrypt uruchamia się z terminala lub wiersza poleceń.

**Podstawowe użycie**

Podaj jako argument ścieżkę do pliku wideo. Skrypt użyje ustawień z pliku `.env`.
```bash
python3 avi2text.py "sciezka/do/mojego_wideo.mp4"

Plik wynikowy mojego_wideo.docx zostanie utworzony w tym samym folderze.

Użycie z flagami (nadpisywanie .env)

Możesz nadpisać domyślne ustawienia za pomocą flag:

--liczba_mowcow: Określa dokładną liczbę mówców w nagraniu.

--model: Wybiera inny model Whisper na czas tego uruchomienia.

--plik_wyjsciowy: Ustala niestandardową nazwę dla pliku .docx.

--jezyk: Zmienia język transkrypcji.

Przykład:

python3 avi2text.py "spotkanie_firmowe.avi" --liczba_mowcow 4 --model medium --plik_wyjsciowy "transkrypcja_spotkania.docx"
