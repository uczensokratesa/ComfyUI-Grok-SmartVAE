# ComfyUI-Grok-SmartVAE
Initial release: Grok's Universal Smart VAE Decode – crash-proof, dynamic batching &amp; force-scale edition.

# 🎬 ComfyUI-Grok-SmartVAE

**Najbardziej odporny i elastyczny dekoder VAE dla ComfyUI**  
(przeznaczony do długich sekwencji wideo: LTX-2, Stable Video Diffusion, CogVideoX, AnimateDiff, etc.)

Ta implementacja łączy najlepsze pomysły z czterech generacji AI:

- **GPT** → solidna baza sliding-window + overlap
- **Gemini** → safety-first + tiling fallback
- **Claude** → matematycznie precyzyjna detekcja skali czasowej (3 klatki + wzór)
- **Grok** → dynamiczne zmniejszanie batcha w locie, force_time_scale, ultra-agresywne OOM recovery

W efekcie powstał node, który jest **blisko crash-proof** – nawet na kartach z 8–12 GB VRAM radzi sobie z długimi filmami 720p/25fps i większymi.

### Główne cechy

- Automatyczna detekcja `time_scale` (lub ręczne wymuszenie: 1, 8, 4…)
- Dynamiczna redukcja rozmiaru batcha przy out-of-memory (z while-loop, nie myli się jak stare for-range)
- Auto-włączanie spatial tiling gdy normalny decode pada
- Inteligentne zszywanie chunków z temporal overlap i spatial crop/align
- Bardzo oszczędne zarządzanie pamięcią (selektywne gc.collect + torch.cuda.empty_cache)
- Obsługuje zarówno obrazy (4D), jak i wideo (5D), multi-batch (rzadkie)

### Instalacja

1. W folderze custom_nodes:
2. git clone https://github.com/uczensokratesa/ComfyUI-Grok-SmartVAE.git

2. Zrestartuj ComfyUI

Node pojawi się w kategorii: **latent/video** → **Grok Universal Smart VAE Decode**

### Porównanie z poprzednikami

| Model    | Detekcja skali | Force scale | Dynamic batch reduction | Auto-tiling on OOM | Pętla     | Ocena stabilności |
|----------|----------------|-------------|--------------------------|---------------------|-----------|-------------------|
| GPT      | podstawowa     | ✗           | ✗                        | ✗                   | for       | ★★☆☆☆            |
| Gemini   | dobra          | ✗           | częściowa                | ✓                   | for       | ★★★★☆            |
| Claude   | bardzo precyzyjna | ✗        | ✗                        | ✓                   | for       | ★★★★☆            |
| **Grok** | bardzo precyzyjna | **✓**    | **pełna (while)**        | **agresywna**       | **while** | **★★★★★**        |

### Historia – rywalizacja i współpraca AI

Cała ta ewolucja zaczęła się od prostego zadania: napisać niezawodny VAE Decode dla workflow z LTX-2.

- GPT dał pierwszą działającą wersję
- Gemini dodał tiling i lepsze OOM handling
- Claude wprowadził najdokładniejszą detekcję skali (3 klatki + równanie)
- Grok dodał force_time_scale i – co najważniejsze – **prawdziwie dynamiczną pętlę while**, która pozwala zmniejszać batch w trakcie dekodowania bez rozsynchronizowania chunków

To jeden z najciekawszych przykładów, jak cztery różne modele mogą się nawzajem poprawiać i budować coś lepszego niż którykolwiek z osobna. Dziękuję @Gemini, @Claude, @GPT i całemu zespołowi xAI za inspirację!

### Licencja

MIT – róbcie z tym co chcecie, tylko zostawcie autora oryginalnego pomysłu (i dajcie znać jeśli zrobicie z tego coś jeszcze lepszego 💪)

Miłego generowania!

---
Stworzone przy współpracy z Grokiem (xAI) – styczeń 2026
