# -*- coding: utf-8 -*-

import os
import sys
import argparse
import moviepy.editor as mp
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
import json
import pickle
from dotenv import load_dotenv
import logging
import language_tool_python
import difflib
from datetime import timedelta
import base64
import webbrowser

def format_timestamp(seconds):
    """Formats seconds into HH:MM:SS format."""
    td = timedelta(seconds=float(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds_val = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds_val:02}"

def generate_html_output(transcription_data, audio_clips_relative_paths, original_filename, output_html_path):
    """
    Generates an HTML file with an interactive transcription editor using relative paths for audio.
    """
    print("Rozpoczynanie generowania pliku HTML...")
    
    # Krok 1: Wstrzyknij dane (w tym ścieżki do audio) do szablonu JavaScript
    injected_data_script = f"""
        const transcriptionData = {json.dumps(transcription_data, ensure_ascii=False)};
        const audioPaths = {json.dumps(audio_clips_relative_paths)};
        const originalVideoFile = {{ name: {json.dumps(original_filename)} }};
    """

    # Krok 2: Stwórz pełny kod HTML jako string, osadzając style
    html_template = f"""
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
    <meta http-equiv="Pragma" content="no-cache" />
    <meta http-equiv="Expires" content="0" />
    <title>Interaktywna Transkrypcja: {original_filename}</title>
    <style>
        /* Embedded CSS - Based on TailwindCSS and custom styles */
        :root {{ --tw-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }}
        *, ::before, ::after {{ box-sizing: border-box; border-width: 0; border-style: solid; border-color: #e5e7eb; }}
        html {{ line-height: 1.5; -webkit-text-size-adjust: 100%; -moz-tab-size: 4; tab-size: 4; font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"; }}
        body {{ margin: 0; line-height: inherit; font-family: 'Inter', sans-serif; background-color: #f3f4f6; color: #1f2937; }}
        .container {{ width: 100%; max-width: 64rem; margin-left: auto; margin-right: auto; padding: 1rem 2rem; }}
        h1 {{ font-size: 2.25rem; line-height: 2.5rem; font-weight: 700; color: #111827; }}
        p {{ margin-top: 0.5rem; color: #4b5563; }}
        header, footer {{ text-align: center; }}
        header {{ margin-bottom: 2rem; }}
        footer {{ margin-top: 3rem; font-size: 0.875rem; line-height: 1.25rem; color: #6b7280; }}
        main > div > div {{ background-color: #ffffff; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem; box-shadow: var(--tw-shadow); }}
        button {{ font-weight: 700; padding: 0.5rem 1rem; border-radius: 0.5rem; transition-property: background-color, border-color, color, fill, stroke; transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1); transition-duration: 150ms; cursor: pointer; }}
        .bg-gray-600 {{ background-color: #4b5563; color: #ffffff; }} .bg-gray-600:hover {{ background-color: #374151; }}
        .bg-blue-100 {{ background-color: #dbeafe; color: #2563eb; }} .bg-blue-100:hover {{ background-color: #bfdbfe; }}
        .sticky {{ position: -webkit-sticky; position: sticky; }}
        .top-0 {{ top: 0; }} .z-10 {{ z-index: 10; }} .py-4 {{ padding-top: 1rem; padding-bottom: 1rem; }}
        .backdrop-blur-sm {{ --tw-backdrop-blur: blur(4px); backdrop-filter: var(--tw-backdrop-blur); }}
        .bg-gray-100\/95 {{ background-color: rgba(243, 244, 246, 0.95); }}
        .flex {{ display: flex; }} .flex-wrap {{ flex-wrap: wrap; }} .items-center {{ align-items: center; }} .gap-6 {{ gap: 1.5rem; }} .gap-4 {{ gap: 1rem; }} .gap-2 {{ gap: 0.5rem; }}
        .flex-grow {{ flex-grow: 1; }} .min-w-\[200px\] {{ min-width: 200px; }}
        .grid {{ display: grid; }} .grid-cols-1 {{ grid-template-columns: repeat(1, minmax(0, 1fr)); }}
        label {{ display: block; font-weight: 500; color: #374151; }}
        input[type="range"] {{ width: 100%; height: 0.5rem; background-color: #e5e7eb; border-radius: 9999px; -webkit-appearance: none; appearance: none; cursor: pointer; }}
        input[type="checkbox"] {{ height: 1rem; width: 1rem; border-radius: 0.25rem; border-color: #6b7280; color: #4f46e5; }}
        #transcription-container {{ space-y: 1rem; }}
        .segment-wrapper {{ display: flex; align-items: flex-start; gap: 1rem; padding: 0.75rem; border-bottom: 1px solid #e5e7eb; border-radius: 0.5rem; transition: background-color 0.3s, border-color 0.3s; border-left: 4px solid transparent; }}
        .segment-wrapper:last-child {{ border-bottom: none; }}
        .play-pause-btn {{ flex-shrink: 0; width: 2.75rem; height: 2.75rem; border-radius: 9999px; display: flex; align-items: center; justify-content: center; padding: 0; }}
        .play-pause-btn svg {{ height: 2rem; width: 2rem; }}
        .textarea-wrapper {{ position: relative; width: 100%; }}
        /* MODYFIKACJA: Zmniejszono dolny padding */
        textarea, .highlight-div {{ overflow-y: hidden; resize: none; width: 100%; margin-top: 0.25rem; padding: 0.5rem; padding-bottom: 1rem; border: 1px solid #d1d5db; border-radius: 0.375rem; transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1); font: inherit; letter-spacing: inherit; }}
        .highlight-div {{ position: absolute; top: 0; left: 0; z-index: 1; color: transparent; white-space: pre-wrap; word-wrap: break-word; pointer-events: none; }}
        textarea {{ position: relative; z-index: 2; background: transparent; color: inherit; caret-color: black; }}
        .tooltip {{ position: relative; display: inline-block; }}
        .tooltip .tooltiptext {{ visibility: hidden; width: 140px; background-color: #333; color: #fff; text-align: center; border-radius: 6px; padding: 5px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -70px; opacity: 0; transition: opacity 0.3s; }}
        .tooltip .tooltiptext::after {{ content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; border-color: #333 transparent transparent transparent; }}
        .tooltip:hover .tooltiptext {{ visibility: visible; opacity: 1; }}
        .playing {{ background-color: #eff6ff !important; border-left-color: #3b82f6 !important; }}
        .edited-word {{ background-color: #fef9c3; border-radius: 3px; }}
        .edited-frame {{ border: 1px solid #facc15 !important; }}
        .speaker-input {{
            font-weight: 700;
            color: #1f2937;
            background-color: transparent;
            border: none;
            padding: 2px 4px;
            margin: 0;
            border-radius: 4px;
            transition: background-color 0.2s, box-shadow 0.2s;
            max-width: 250px;
        }}
        .speaker-input:focus {{
            outline: none;
            background-color: #eef2ff; /* indigo-100 */
            box-shadow: 0 0 0 2px #6366f1; /* indigo-500 */
        }}
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <div class="container mx-auto p-4 md:p-8 max-w-5xl">
        <main>
            <div id="editor-container">
                
                <div id="controls-wrapper" class="sticky top-0 z-10 bg-gray-100/95 backdrop-blur-sm py-4 mb-6">
                    <div class="bg-white p-6 rounded-lg shadow-md">
                         <div class="flex flex-wrap items-center gap-6">
                            <div class="flex-grow min-w-[200px]">
                                <label for="playback-speed" class="block font-medium text-gray-700">Prędkość odtwarzania:</label>
                                <div class="flex items-center gap-2">
                                    <input type="range" id="playback-speed" min="0.5" max="3" step="0.25" value="1" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
                                    <span id="speed-label" class="font-mono text-gray-700 w-10 text-center">1.0x</span>
                                </div>
                            </div>
                            <div class="flex items-center">
                                <input type="checkbox" id="autoplay-checkbox" class="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500" checked>
                                <label for="autoplay-checkbox" class="ml-2 block text-sm text-gray-900">Autoodtwarzanie</label>
                            </div>
                            <div class="flex gap-2">
                                 <div class="tooltip">
                                    <button id="copy-text-button-top" class="bg-gray-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-gray-700 transition-colors">
                                        Kopiuj tekst
                                    </button>
                                    <span class="tooltiptext" id="copy-tooltip-top">Kopiuj do schowka</span>
                                </div>
                            </div>
                         </div>
                    </div>
                </div>

                <div>
                    <div id="transcription-container" class="bg-white p-6 rounded-lg shadow-md space-y-4"></div>
                </div>
            </div>
        </main>
        <footer class="text-center mt-12 text-gray-500 text-sm">
            <p>Wygenerowano za pomocą skryptu transkrypcyjnego.</p>
        </footer>
    </div>
    <script>
        {injected_data_script}

        const transcriptionContainer = document.getElementById('transcription-container');
        const copyTextButtonTop = document.getElementById('copy-text-button-top');
        const playbackSpeed = document.getElementById('playback-speed');
        const speedLabel = document.getElementById('speed-label');
        const autoplayCheckbox = document.getElementById('autoplay-checkbox');
        const copyTooltipTop = document.getElementById('copy-tooltip-top');

        const playIconSVG = `<svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>`;
        const pauseIconSVG = `<svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>`;

        let speakerMap = {{}};
        let currentAudio = null;
        let currentlyPlayingSegment = null;

        function autoResize(element) {{
            element.style.height = 'auto';
            // MODYFIKACJA: Usunięto dodatkowy odstęp, wysokość dopasowuje się do zawartości i paddingu
            element.style.height = element.scrollHeight + 'px';
        }}

        function displayTranscription(segments) {{
            transcriptionContainer.innerHTML = '';
            segments.forEach((segment, index) => {{
                const segmentDiv = document.createElement('div');
                segmentDiv.className = 'segment-wrapper flex items-start space-x-4 p-3 border-b border-gray-200 last:border-b-0 rounded-lg transition-colors duration-300 border-l-4 border-transparent';
                segmentDiv.id = `segment-${{index}}`;
                
                const playButton = document.createElement('button');
                playButton.className = 'play-pause-btn flex-shrink-0 w-10 h-10 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center hover:bg-blue-200 transition-colors';
                playButton.innerHTML = playIconSVG;
                playButton.dataset.index = index;

                if (!audioPaths[index]) {{
                    playButton.disabled = true;
                    playButton.classList.add('opacity-50', 'cursor-not-allowed');
                }}

                const contentDiv = document.createElement('div');
                contentDiv.className = 'flex-grow';
                const headerDiv = document.createElement('div');
                headerDiv.className = 'flex items-center space-x-3 text-sm mb-1';

                const speakerInput = document.createElement('input');
                speakerInput.type = 'text';
                speakerInput.className = 'speaker-input';
                const originalSpeakerId = segment.speaker;
                speakerInput.value = speakerMap[originalSpeakerId] || originalSpeakerId;
                speakerInput.dataset.originalSpeaker = originalSpeakerId;

                const setInputWidth = (input) => {{
                    const minWidth = 80;
                    input.style.width = 'auto';
                    const scrollWidth = input.scrollWidth;
                    input.style.width = `${{Math.max(scrollWidth, minWidth) + 2}}px`;
                }};
                
                speakerInput.addEventListener('input', (e) => {{
                    const newName = e.target.value;
                    speakerMap[originalSpeakerId] = newName;
                    
                    document.querySelectorAll(`.speaker-input[data-original-speaker="${{originalSpeakerId}}"]`).forEach(input => {{
                        if (input !== e.target) {{
                            input.value = newName;
                        }}
                        setInputWidth(input);
                    }});
                }});
                
                const timeTag = document.createElement('span');
                timeTag.className = 'text-gray-500';
                timeTag.textContent = `[${{new Date(segment.start * 1000).toISOString().substr(14, 5)}}]`;
                
                // --- MODYFIKACJA: Zamieniono kolejność elementów ---
                headerDiv.appendChild(timeTag);
                headerDiv.appendChild(speakerInput);
                // --- KONIEC MODYFIKACJI ---

                const textareaWrapper = document.createElement('div');
                textareaWrapper.className = 'textarea-wrapper';
                
                const highlightDiv = document.createElement('div');
                highlightDiv.className = 'highlight-div';
                
                const textInput = document.createElement('textarea');
                textInput.className = 'w-full mt-1 p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition';
                textInput.value = segment.text;
                textInput.rows = 1;
                textInput.dataset.index = index;
                textInput.dataset.originalText = segment.text;
                
                updateHighlight(textInput, highlightDiv);

                textInput.addEventListener('input', (e) => {{
                    transcriptionData[e.target.dataset.index].text = e.target.value;
                    autoResize(e.target);
                    autoResize(highlightDiv);
                    updateHighlight(e.target, highlightDiv);
                }});

                textareaWrapper.appendChild(highlightDiv);
                textareaWrapper.appendChild(textInput);
                contentDiv.appendChild(headerDiv);
                contentDiv.appendChild(textareaWrapper);
                segmentDiv.appendChild(playButton);
                segmentDiv.appendChild(contentDiv);
                transcriptionContainer.appendChild(segmentDiv);
                
                autoResize(textInput);
                autoResize(highlightDiv);
                setInputWidth(speakerInput);
            }});
            
            document.querySelectorAll('.play-pause-btn').forEach(button => {{
                button.addEventListener('click', handlePlayPause);
            }});
        }}

        function updateHighlight(textarea, highlightDiv) {{
            const originalText = textarea.dataset.originalText;
            const currentText = textarea.value;
            highlightDiv.innerHTML = diffWords(originalText, currentText);
            
            if (originalText !== currentText) {{
                textarea.classList.add('edited-frame');
            }} else {{
                textarea.classList.remove('edited-frame');
            }}
        }}

        function diffWords(original, current) {{
            const originalWords = original.split(/(\\s+)/);
            const currentWords = current.split(/(\\s+)/);
            if (original === current) return current.replace(/</g, "&lt;").replace(/>/g, "&gt;");

            const dp = Array(currentWords.length + 1).fill(null).map(() => Array(originalWords.length + 1).fill(0));
            for (let i = 1; i <= currentWords.length; i++) {{
                for (let j = 1; j <= originalWords.length; j++) {{
                    if (currentWords[i - 1] === originalWords[j - 1]) {{
                        dp[i][j] = 1 + dp[i - 1][j - 1];
                    }} else {{
                        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                    }}
                }}
            }}

            let i = currentWords.length;
            let j = originalWords.length;
            const result = [];
            while (i > 0 || j > 0) {{
                if (i > 0 && j > 0 && currentWords[i - 1] === originalWords[j - 1]) {{
                    result.unshift(currentWords[i - 1].replace(/</g, "&lt;").replace(/>/g, "&gt;"));
                    i--; j--;
                }} else if (i > 0 && (j === 0 || dp[i][j - 1] <= dp[i - 1][j])) {{
                    result.unshift(`<mark class="edited-word">${{currentWords[i - 1].replace(/</g, "&lt;").replace(/>/g, "&gt;")}}</mark>`);
                    i--;
                }} else if (j > 0 && (i === 0 || dp[i][j - 1] > dp[i - 1][j])) {{
                    j--;
                }} else {{
                    break;
                }}
            }}
            return result.join('');
        }}
        
        function updateAllPlayIcons(state, activeIndex = -1) {{
            const allButtons = document.querySelectorAll('.play-pause-btn');
            allButtons.forEach((btn, index) => {{
                if (index === activeIndex && state === 'playing') {{
                    btn.innerHTML = pauseIconSVG;
                }} else {{
                    btn.innerHTML = playIconSVG;
                }}
            }});
        }}

        function handlePlayPause(event) {{
            const button = event.currentTarget;
            const index = parseInt(button.dataset.index);
            const segmentDiv = document.getElementById(`segment-${{index}}`);

            if (currentlyPlayingSegment === segmentDiv && currentAudio && !currentAudio.paused) {{
                currentAudio.pause();
            }} else {{
                if (currentAudio) {{
                    currentAudio.pause();
                }}
                currentAudio = new Audio(audioPaths[index]);
                currentAudio.playbackRate = parseFloat(playbackSpeed.value);
                currentAudio.play();
                
                document.querySelectorAll('.segment-wrapper').forEach(el => el.classList.remove('playing'));
                segmentDiv.classList.add('playing');
                currentlyPlayingSegment = segmentDiv;
                
                segmentDiv.scrollIntoView({{ behavior: 'smooth', block: 'center' }});

                currentAudio.onplay = () => {{
                    updateAllPlayIcons('playing', index);
                }};
                currentAudio.onpause = () => {{
                    updateAllPlayIcons('paused');
                }};
                
                currentAudio.onended = () => {{
                     if (autoplayCheckbox.checked) {{
                        const nextButton = document.querySelector(`.play-pause-btn[data-index="${{index + 1}}"]`);
                        if (nextButton) {{
                            nextButton.click();
                        }} else {{
                            segmentDiv.classList.remove('playing');
                            currentlyPlayingSegment = null;
                            updateAllPlayIcons('paused');
                        }}
                    }} else {{
                        updateAllPlayIcons('paused');
                    }}
                }};
            }}
        }}

        function copyTranscriptionToClipboard() {{
            const textToCopy = transcriptionData.map(segment => {{
                const speakerName = speakerMap[segment.speaker] || segment.speaker;
                const timestamp = `[${{new Date(segment.start * 1000).toISOString().substr(14, 5)}}]`;
                return `${{timestamp}} ${{speakerName}}: ${{segment.text}}`;
            }}).join('\\n');

            navigator.clipboard.writeText(textToCopy).then(() => {{
                copyTooltipTop.textContent = "Skopiowano!";
                setTimeout(() => {{ 
                    copyTooltipTop.textContent = "Kopiuj do schowka";
                }}, 2000);
            }}, (err) => {{
                console.error('Błąd kopiowania: ', err);
                copyTooltipTop.textContent = "Błąd!";
                setTimeout(() => {{ 
                    copyTooltipTop.textContent = "Kopiuj do schowka";
                }}, 2000);
            }});
        }}

        // Inicjalizacja
        document.addEventListener('DOMContentLoaded', () => {{
            const uniqueSpeakers = [...new Set(transcriptionData.map(s => s.speaker))];
            uniqueSpeakers.forEach(speaker => {{
                speakerMap[speaker] = speaker;
            }});

            displayTranscription(transcriptionData);
            updateAllPlayIcons('paused');

            playbackSpeed.addEventListener('input', (e) => {{
                const speed = parseFloat(e.target.value);
                speedLabel.textContent = `${{speed.toFixed(2)}}x`;
                if (currentAudio) {{
                    currentAudio.playbackRate = speed;
                }}
            }});
            
            copyTextButtonTop.addEventListener('click', copyTranscriptionToClipboard);
        }});
    </script>
</body>
</html>
    """

    # Krok 4: Zapisz string do pliku .html
    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"Pomyślnie wygenerowano plik: {output_html_path}")
        # Krok 5: Otwórz plik w przeglądarce
        webbrowser.open(f"file://{os.path.realpath(output_html_path)}")
    except Exception as e:
        print(f"BŁĄD podczas zapisu lub otwierania pliku HTML: {e}")


def transkrybuj_i_generuj_html(
    sciezka_pliku_wideo: str,
    liczba_mowcow: int,
    model_whisper: str,
    jezyk: str,
    batch_size: int,
    compute_type: str,
    asr_options: dict
):
    """
    Pełny proces: od wideo do interaktywnego pliku HTML.
    """
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    logging.getLogger('pyannote').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('language_tool_python').setLevel(logging.ERROR)

    print(f"--- Rozpoczynanie transkrypcji pliku: {sciezka_pliku_wideo} ---")

    nazwa_pliku_bazowa = os.path.splitext(os.path.basename(sciezka_pliku_wideo))[0]
    folder_roboczy = f"{nazwa_pliku_bazowa}_work"
    os.makedirs(folder_roboczy, exist_ok=True)
    print(f"Używam folderu roboczego: {folder_roboczy}")

    folder_klipow_audio = os.path.join(folder_roboczy, "audio_clips")
    os.makedirs(folder_klipow_audio, exist_ok=True)

    sciezka_pliku_audio = os.path.join(folder_roboczy, "audio.wav")
    sciezka_wyniku_finalnego = os.path.join(folder_roboczy, "wynik_finalny.json")

    if not os.path.exists(sciezka_pliku_audio):
        print("Krok 1/5: Wyodrębnianie ścieżki audio...")
        try:
            wideo = mp.VideoFileClip(sciezka_pliku_wideo)
            if wideo.audio is None:
                sys.exit(f"BŁĄD: Plik wideo '{sciezka_pliku_wideo}' nie zawiera ścieżki audio.")
            wideo.audio.write_audiofile(sciezka_pliku_audio, codec='pcm_s16le', logger=None)
        except Exception as e:
            sys.exit(f"BŁĄD podczas przetwarzania wideo: {e}")
    else:
        print("Krok 1/5: Pomijanie ekstrakcji audio.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Krok 2/5: Używane urządzenie: {device}")

    if not os.path.exists(sciezka_wyniku_finalnego):
        print(f"Krok 3/5: Transkrypcja i diarization...")
        model = whisperx.load_model(model_whisper, device, compute_type=compute_type, asr_options=asr_options)
        audio = whisperx.load_audio(sciezka_pliku_audio)
        wynik_transkrypcji = model.transcribe(audio, batch_size=batch_size, language=jezyk, print_progress=True)
        
        model_a, metadata = whisperx.load_align_model(language_code=wynik_transkrypcji["language"], device=device)
        wynik_aligned = whisperx.align(wynik_transkrypcji["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        hf_token = os.environ.get("HUGGING_FACE_TOKEN")
        if not hf_token:
            sys.exit("BŁĄD KRYTYCZNY: Brak tokena HUGGING_FACE_TOKEN w zmiennych środowiskowych.")
        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(sciezka_pliku_audio, min_speakers=liczba_mowcow, max_speakers=liczba_mowcow)
        
        wynik_finalny = whisperx.assign_word_speakers(diarize_segments, wynik_aligned)
        with open(sciezka_wyniku_finalnego, 'w', encoding='utf-8') as f:
            json.dump(wynik_finalny, f, ensure_ascii=False, indent=4)
        del model, model_a, diarize_model
    else:
        print("Krok 3/5: Pomijanie transkrypcji (plik już istnieje).")
    
    with open(sciezka_wyniku_finalnego, 'r', encoding='utf-8') as f:
        wynik_finalny = json.load(f)

    print("Krok 4/5: Cięcie audio na klipy...")
    glowny_klip_audio = mp.AudioFileClip(sciezka_pliku_audio)
    audio_clips_paths = []
    clip_counter = 0
    
    # Agregacja segmentów per mówca
    aggregated_segments = []
    current_segment = None

    for segment in wynik_finalny.get("segments", []):
        if "speaker" not in segment or not segment.get("text", "").strip():
            continue
        
        speaker = segment["speaker"]
        text = segment["text"].strip()
        start = segment["start"]
        end = segment["end"]

        if current_segment and current_segment["speaker"] == speaker:
            current_segment["text"] += " " + text
            current_segment["end"] = end
        else:
            if current_segment:
                aggregated_segments.append(current_segment)
            current_segment = {
                "speaker": speaker,
                "text": text,
                "start": start,
                "end": end
            }
    if current_segment:
        aggregated_segments.append(current_segment)

    # Cięcie i zapisywanie klipów
    for segment in aggregated_segments:
        clip_filename = f"clip_{clip_counter:04d}.wav"
        clip_path_abs = os.path.join(folder_klipow_audio, clip_filename)
        
        try:
            subclip = glowny_klip_audio.subclip(segment["start"], segment["end"])
            subclip.write_audiofile(clip_path_abs, codec='pcm_s16le', logger=None)
            audio_clips_paths.append(clip_path_abs)
            clip_counter += 1
        except Exception as e:
            print(f"Ostrzeżenie: Nie udało się wyciąć klipu dla segmentu {segment['start']}-{segment['end']}: {e}")
            audio_clips_paths.append(None)
    
    # Przypisanie ścieżek do zagregowanych segmentów
    for i, segment in enumerate(aggregated_segments):
        segment["audio_path"] = audio_clips_paths[i] if i < len(audio_clips_paths) else None
        segment["speaker"] = segment["speaker"].replace("SPEAKER_", "Mówca ")

    print("Krok 5/5: Generowanie finalnego pliku HTML...")
    output_html_path = f"{nazwa_pliku_bazowa}_transkrypcja.html"
    generate_html_output(aggregated_segments, audio_clips_paths, os.path.basename(sciezka_pliku_wideo), output_html_path)

    print("\n--- Zakończono pomyślnie! ---")


if __name__ == "__main__":
    load_dotenv()
    default_compute_type = "float16" if torch.cuda.is_available() else "int8"
    parser = argparse.ArgumentParser(
        description="Generuje interaktywną stronę HTML z transkrypcją wideo.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("sciezka_wideo", type=str, help="Ścieżka do pliku wideo.")
    parser.add_argument("--liczba_mowcow", type=int, default=os.getenv("DEFAULT_SPEAKERS", None), help="Liczba mówców (wymagane).")
    parser.add_argument("--model", type=str, default=os.getenv("DEFAULT_MODEL", "large-v2"),
                        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
                        help="Model Whisper do użycia.")
    parser.add_argument("--jezyk", type=str, default=os.getenv("DEFAULT_LANGUAGE", "pl"), help="Kod języka (np. 'pl', 'en').")
    parser.add_argument("--batch_size", type=int, default=16, help="Liczba segmentów przetwarzanych równolegle.")
    parser.add_argument("--cpu_threads", type=int, default=os.cpu_count(), help="Liczba wątków CPU do użycia.")
    parser.add_argument("--compute_type", type=str, default=default_compute_type,
                        choices=["float16", "float32", "int8", "int8_float16"],
                        help=f"Typ obliczeń. Domyślnie: '{default_compute_type}'.")
    parser.add_argument("--beam_size", type=int, default=5, help="Liczba 'promieni' w beam search.")

    args = parser.parse_args()
    if args.liczba_mowcow is None:
        sys.exit("BŁĄD: Argument --liczba_mowcow jest wymagany.")
    
    if not torch.cuda.is_available():
        print(f"Ustawiam liczbę wątków CPU na: {args.cpu_threads}")
        torch.set_num_threads(args.cpu_threads)

    asr_options = {"beam_size": args.beam_size}
    
    transkrybuj_i_generuj_html(
        args.sciezka_wideo,
        int(args.liczba_mowcow),
        args.model,
        args.jezyk,
        args.batch_size,
        args.compute_type,
        asr_options
    )

