# AI Browser Agent

A Chrome side-panel AI assistant for normal HTTP and HTTPS websites. It reads the
real DOM, summarizes pages, searches using observed inputs, follows links, and answers
questions about selected PDFs with citations. Ollama runs the default 2B router and
configurable main model locally. Cloud provider credentials stay on the backend.

**Quick path:** install/setup -> start backend -> load extension -> open website ->
click extension -> use assistant. Pair the extension with the generated backend token
once in Settings. No domain allowlist or site-access approval is needed to start.

See [architecture and limitations](docs/architecture.md) and [testing](docs/testing.md). This is a local developer setup, not a published Chrome Web Store package.

## What you need

- Windows with PowerShell, [Git](https://git-scm.com/downloads/win),
  [Python](https://www.python.org/downloads/windows/),
  [Node.js](https://nodejs.org/en/download), and Google Chrome.
- Use **Python 3.14** and [**Node.js 24 LTS**](https://nodejs.org/en/about/previous-releases) for the documented Windows setup.
  Node 22.12 or newer is required by the build tools. The development workstation
  uses Python 3.14.2 and Node 24.11.1.
- [Ollama for Windows](https://docs.ollama.com/windows).
- Enough local disk for the Python environment and several large model downloads.
  A CUDA-capable NVIDIA GPU helps. The development machine has an RTX 2000 Ada
  with 16 GB VRAM; this is not a guarantee that all models fit in GPU memory at once.

Use the 64-bit installers. Enable Python on PATH during installation and reopen
PowerShell after installing these tools.

Installation downloads packages and public models. After installation, the local
AI profiles do not need a cloud AI service. Website access still requires a network connection.
Sign in manually when a website needs authentication; password fields are blocked from agent input.

## Models used

| Model | Role | Needed for |
| --- | --- | --- |
| `qwen3.5:2b` in Ollama | Chooses the browser tool group | Normal browser questions |
| `qwen3.5:9b` in Ollama | Chooses actions and writes answers; accepts images | Normal browser questions and PDF answers |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Local ONNX text embeddings, combined with BM25 | PDF text retrieval |
| `nomic-ai/nomic-embed-multimodal-3b` | Local PDF page-image embeddings | PDF visual retrieval |
| `Qwen/Qwen2.5-VL-3B-Instruct` | Base weights required by the Nomic adapter | PDF visual retrieval |

**Installing Python packages does not download any models.**
`pip install -r requirements.txt` installs the software that runs them. Download
the model files separately:

| What you want to do | Downloads needed |
| --- | --- |
| Browse and ask questions about websites | Both Ollama Qwen models in step 3 |
| Also ask questions about PDF text | Step 3 plus MiniLM in step 6 |
| Also retrieve PDF images, tables and diagrams | Steps 3 and 6 plus Nomic **and its Qwen base** in step 7 |

The Qwen base in step 7 is different from the two Ollama chat models. Downloading
one does not install the others. Downloads need internet access and can take time.
You normally download the models only once per computer.

Nomic is loaded by the Python visual worker, **not by Ollama**. It does not replace
the chat model. Text and visual results are combined using reciprocal rank fusion;
there is no separate learned reranker in the default pipeline.

## 1. Open the project and install requirements

Clone the repository using the URL supplied by your project maintainer, or open
an existing checkout or source snapshot. Git metadata is optional; root detection uses
`backend/`, `extension/` and `shared/build-contract.json`. Git may still be required
to install source dependencies. In PowerShell, change to the project root, where `README.md`
and `scripts/` are located. Prefer a local disk folder, such as
`C:\work\ai-browser-agent`. If using a network checkout, use its mapped
drive path (for example `U:\ai-browser-agent`), rather than a UNC path
with npm on Windows.

Check that the tools are available:

```powershell
python --version
node --version
git --version
ollama --version
```

The Python requirements file is [requirements.txt](requirements.txt) in the
project root. It includes the pinned package list from
[backend/requirements.txt](backend/requirements.txt), so both installation methods
use the same dependencies. Git must be installed because one package is installed
from its source repository.

Run these commands **one at a time**, from the project root. Continue only when
each command finishes successfully.

**Create and activate the Python environment:**

```powershell
python -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1
```

Your terminal should now show `(.venv)`.

**Install the Python requirements:**

```powershell
pip install -r requirements.txt
python -m pip check
```

The `-r` tells pip to read the packages from the requirements file. This installs
the backend and PDF retrieval libraries. **Model weights are downloaded separately
in steps 3, 6, and 7 below.**

**Install and build the Chrome extension:**

```powershell
Push-Location extension
npm.cmd ci
Pop-Location
.\scripts\build.ps1
```

The extension is now built in `extension/dist`. Continue with step 2 to configure
the backend, then follow the remaining steps to download models and connect Chrome.

Alternatively, `.\scripts\setup.ps1` performs the environment creation, Python
installation, extension dependency installation, and build above automatically.
Choose either the manual commands or this setup script; you do not need both.
Neither method overwrites your existing `.env` or downloads model weights.

If PowerShell blocks local scripts and your workstation policy permits it, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Then retry the blocked command. Later commands in this guide use the environment's
full Python path, so they also work after you open a new terminal without activation.

## 2. Create your private configuration

```powershell
.\backend\.venv\Scripts\python.exe -B tools/init_local_config.py
```

This creates `backend/.env` with a random backend connection token. An existing
file is preserved. Open it privately in your editor and keep this selection:

```dotenv
BROWSER_PROVIDER__ACTIVE=local_agent
```

Keep the generated `BROWSER_AUTH__TOKEN` value: you will copy it into the extension
Settings in step 5. Do not paste the token into chat, logs, or version control.
You can leave cloud-provider credentials empty.

The defaults in `backend/config.yaml` already configure both Ollama models on
`http://127.0.0.1:11434`. To override the main model, use the **LOCAL_AGENT** profile:

```dotenv
BROWSER_PROVIDERS__LOCAL_AGENT__MODEL=qwen3.5:9b
BROWSER_PROVIDERS__LOCAL_AGENT__BASE_URL=http://127.0.0.1:11434
BROWSER_PROVIDERS__LOCAL_ROUTER__MODEL=qwen3.5:2b
```

Restart the backend after changing `.env` or `config.yaml`. An extension rebuild
is not needed for those changes.

## 3. Start Ollama and download the two chat models

Ollama's Windows app may already be running in the notification area. Use one
server instance. For explicit local-only settings, quit that app, open a separate
PowerShell terminal, and run:

```powershell
$env:OLLAMA_NO_CLOUD = "1"
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
ollama serve
```

Leave this terminal open. These settings belong to the **Ollama server process**,
not to `backend/.env`. Alternatively, configure Windows user environment variables
and restart the Ollama app, as described in the
[Ollama Windows guide](https://docs.ollama.com/windows).

In a **second PowerShell terminal**, open the project root (the folder containing
this README). Run these commands one at a time. Wait for each download to finish
successfully before running the next command:

```powershell
# Download the small router model.
ollama pull qwen3.5:2b

# Download the main assistant model.
ollama pull qwen3.5:9b

# Check that both models are installed.
ollama list
.\backend\.venv\Scripts\python.exe -B tools/local_routing_check.py --run
```

Both model names should appear in `ollama list`. The routing check makes actual
local model calls with synthetic input. It uses synthetic data and does not browse websites.
The first request can take longer while model weights load.

## 4. Check and start the backend

```powershell
.\scripts\doctor.ps1
.\scripts\start-backend.ps1
```

Doctor checks configuration, installed dependencies, local storage, and build
compatibility. It does **not** prove that an AI model can answer correctly.
If it reports missing MiniLM artifacts, complete step 6 and run Doctor again.
Leave the backend terminal running. Successful startup reports
`active provider type: ollama` and `backend ready` on loopback port `8765`.

## 5. Load and connect the Chrome extension

1. Open `chrome://extensions` and enable **Developer mode**.
2. Select **Load unpacked** and choose the project's **`extension/dist`** folder.
3. Open the extension's **Settings**.
4. Set **Backend URL** to `ws://127.0.0.1:8765` (do not add `/ws`).
5. Paste the value of `BROWSER_AUTH__TOKEN` from your private `backend/.env`
   into **Token**. Select **Save**, then **Test connection**.
6. Open any normal HTTP/HTTPS website and click the extension icon.
7. Ask `Summarize this page` or `Search this site for installation instructions`.

The panel should show **Backend connected - ollama; protocol compatible**.
Try `Summarize this page`, then a question that needs a source page, such as
`Find the installation instructions on this site`.

Routine steps run automatically. The shipped `delete_only` policy requests
confirmation for recognized deletion actions. The assistant remains tied to its
own tab; Pause, Resume, and Stop remain available. It must follow observed links
and cannot bypass Chrome or website access controls. There is no Read-only mode.

Chrome grants temporary access when you click the extension. After crossing to another
origin, click the extension again if Chrome revokes access. Optional persistent site
access is available in Settings; enter a site URL and grant only the sites you choose.
Chrome/internal/extension/file pages and Chrome-protected pages cannot be controlled.
The assistant stays bound to its owner tab; a newly opened tab has its own assistant.

## 6. Install the PDF text embedding model

**This downloads only MiniLM, the PDF text embedding model. It does not download
Nomic or any Qwen model.** Skip it if you only want website browsing.

After installing the requirements in step 1, open PowerShell at the project root.
Run the first command below and wait for it to finish. Then run the second command
to check the installed artifacts:

```powershell
.\backend\.venv\Scripts\python.exe -B tools/local_retrieval.py prepare-model
.\backend\.venv\Scripts\python.exe -B tools/local_retrieval.py status
```

This downloads and checks the MiniLM ONNX artifacts at revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Do not substitute an arbitrary model folder:
the project checks the artifact manifest and vector dimensions.

PDF text retrieval uses the CPU ONNX runtime included in the requirements.
Do not install `onnxruntime-gpu` alongside `onnxruntime` in this environment.

## 7. Install Nomic for PDF images, tables, and diagrams

**You must download both Nomic and its Qwen base for visual PDF retrieval.**
This step is optional for website browsing and PDF text retrieval. Python loads
these files directly; they are separate from the Ollama chat models.

Complete steps 1 and 2 first so the Python environment, `hf.exe` download tool,
and private `backend/.env` file exist. Open **PowerShell at the project root**.
Keep that terminal open. Copy the contents of each code block, without the
surrounding Markdown backticks.

**7a. Create the model folder.** Copy and paste this block:

```powershell
$modelRoot = Join-Path $env:LOCALAPPDATA "AIBrowserAgent\models"
New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null
Test-Path .\backend\.venv\Scripts\hf.exe
```

The last command should print `True`. If it prints `False`, complete the requirements
installation in step 1 first. `$modelRoot` uses your Windows account's local folder;
no username replacement is needed. If you open a new terminal, run step 7a again.

**7b. Download Nomic.** Copy the whole command and wait for it to finish:

```powershell
.\backend\.venv\Scripts\hf.exe download nomic-ai/nomic-embed-multimodal-3b --revision 298930bb768c50b91d2799d6f3b0daf46ea52e70 --local-dir "$modelRoot\nomic-embed-multimodal-3b"
```

**7c. Download its required Qwen base.** After 7b succeeds, run:

```powershell
.\backend\.venv\Scripts\hf.exe download Qwen/Qwen2.5-VL-3B-Instruct --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 --local-dir "$modelRoot\Qwen2.5-VL-3B-Instruct"
```

These are large downloads. Wait until each command finishes and the PowerShell
prompt returns without an error. Leave the long revision values unchanged: they
select the project's pinned model versions. If a download fails, resolve the
reported error and rerun that command before continuing.

**7d. Configure the downloaded paths.** Run this block to print three configuration
lines with the correct paths for your computer:

```powershell
$modelRoot = Join-Path $env:LOCALAPPDATA "AIBrowserAgent\models"
$configModelRoot = $modelRoot.Replace('\', '/')
Write-Output "BROWSER_VISUAL_RETRIEVAL__MODEL_DIRECTORY=$configModelRoot/nomic-embed-multimodal-3b"
Write-Output "BROWSER_VISUAL_RETRIEVAL__BASE_MODEL_DIRECTORY=$configModelRoot/Qwen2.5-VL-3B-Instruct"
Write-Output "BROWSER_VISUAL_RETRIEVAL__DEVICE=auto"
```

Open `backend/.env` in your editor and paste those **three printed lines** into the
file. If the same settings already exist, replace their values instead of adding
duplicates. Save the file and leave your existing token and other settings intact.
This block prints only model settings; it does not read or modify `.env`.

**7e. Choose the runtime and verify the models.** Step 1 installs the base Python
runtime. For CPU use, set `BROWSER_VISUAL_RETRIEVAL__DEVICE=cpu` in `.env` and go to
the status check below. For an NVIDIA GPU, follow the CUDA commands next.

For the NVIDIA CUDA runtime, after installing the base requirements:

```powershell
.\backend\.venv\Scripts\python.exe -m pip install --no-deps -r backend/requirements-visual-cuda.txt
.\backend\.venv\Scripts\python.exe -m pip check
.\backend\.venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

The CUDA requirements select matching Torch/Torchvision wheels. A compatible
NVIDIA driver is required. Without a working GPU, `DEVICE=cpu` is available but
large-model indexing is much slower.

For a network checkout or a separate Python installation, create the visual
worker on local disk instead:

```powershell
.\backend\.venv\Scripts\python.exe -B tools/setup_visual_runtime.py --device cuda
```

Set `BROWSER_VISUAL_RETRIEVAL__PYTHON_EXECUTABLE` to the `python_executable` path
returned by that helper. Use `--device cpu` for a CPU worker. This worker has its
own packages; installing CUDA in the backend environment does not update it.

**Check the downloads and configuration.** From the project root, run:

```powershell
.\backend\.venv\Scripts\python.exe -B tools/local_visual_retrieval.py status
```

If the check reports missing or unavailable model artifacts, recheck both downloads
and the two paths in `.env`. That message means setup is not complete.

After saving the configuration, stop a running backend with **Ctrl+C in its own
terminal**, then start it again from the project root:

```powershell
.\scripts\start-backend.ps1
```

`not_indexed` is normal before the first visual PDF preparation. A status check
is not a full GPU inference test. Preparing and asking about a
small PDF is the final functional check. The
[Nomic model card](https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b)
describes its Qwen base and research-use weights; model license review remains
part of production release preparation.

## 8. Ask about a PDF

PDFs that open without a password and allow text extraction are supported,
including permission-encrypted PDFs. The requirements include AES support
(`pypdf[crypto]` and `cryptography`). PDFs that require an opening password or
prohibit extraction need an appropriate copy from the document owner.

1. Find a PDF on a website with an observed direct PDF link.
2. Select **Prepare locally** on its file card. Wait until preparation completes.
3. Select **Ask about this PDF**, then ask a question about that document.
4. Check the cited page references against the original PDF.

To compare several PDFs from search results, explicitly select **Research multiple
PDFs from search results** before sending the question. Leave it off for normal
page questions or when using **Ask about this PDF**.

Preparation downloads the selected attachment using your browser session and
indexes it locally. It does not itself call a remote answer provider. When Nomic
is available, questions combine text and visual evidence automatically. Missing
visual weights allow text retrieval to continue; they do not prove visual indexing
worked. The remote-provider consent checkbox is not required for the configured
local-only Ollama profile.

With a network checkout, SQLite and private staging data use the local Windows
application-data directory automatically. Avoid putting live SQLite databases
on a network share. Keep `.env`, indexed documents, logs, and model caches private.

## Starting again tomorrow

1. Start Ollama (with the same server settings).
2. From the project root, run `.\scripts\start-backend.ps1`.
3. Open Chrome, open your website, and open the assistant.

Do not reinstall requirements or pull models every time. After extension source
changes, run `.\scripts\build.ps1`, click **Reload** in `chrome://extensions`, and
refresh the website tab. After backend-only changes, restart the backend.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Port 8765 occupied | Check whether your backend is already running. Stop that backend with Ctrl+C in its terminal before starting another. The startup script does not kill processes. |
| Extension disconnected | Backend terminal running; URL uses `ws://`; token matches; Settings saved. |
| Protocol/build mismatch | Run `scripts/build.ps1`, reload the extension, then refresh the website tab. |
| Local router unavailable | Run `ollama list` and the routing check from step 3; both configured models must exist. |
| Ollama ignores a setting | Server settings belong to Ollama; backend profile overrides must use `LOCAL_AGENT` or `LOCAL_ROUTER`. Restart the relevant process. |
| `visual_embedding_model_unavailable` | Both Nomic adapter and Qwen base are required. Check their explicit paths and the selected worker's packages. |
| PDF encryption support missing | From the project root, run `.\backend\.venv\Scripts\python.exe -m pip install -r requirements.txt`, restart the backend, and retry preparation. |
| GPU out of memory | Close other GPU workloads. Use `ollama stop qwen3.5:9b` and `ollama stop qwen3.5:2b` before heavy visual indexing, or select the CPU visual worker. Ollama can reload models on the next request. |
| Agent stops or gets stuck | Keep the timeline/error code. Runs still have step/time and repeat bounds. An observed source link is required; guessed URLs are rejected. |
| Link opens in another tab | This remains a known limitation. Stop the old run and ask again in the destination tab's own assistant. |
| No relevant answer | Give the site/location and topic. The agent should inspect a source page; search snippets alone do not verify a contact. |

### Longer timeouts for slow local models

If you see `Ollama request timed out` or `A model step timed out`, add or update
these lines in `backend/.env` (all values are seconds):

```dotenv
BROWSER_PROVIDERS__LOCAL_AGENT__TIMEOUT_SECONDS=900
BROWSER_PROVIDERS__LOCAL_ROUTER__TIMEOUT_SECONDS=300
BROWSER_ROUTING__FALLBACK_TIMEOUT_SECONDS=300
BROWSER_AGENT__STEP_TIMEOUT_SECONDS=1200
BROWSER_AGENT__RUN_TIMEOUT_SECONDS=3600
```

This allows 15 minutes for a main-model request, 5 minutes for routing,
20 minutes per agent step, and 60 minutes for the whole task. Keep the step
timeout longer than the model request, and the whole-task timeout longer than
one step. The earliest applicable timeout ends the operation.

You can also edit the matching settings in `backend/config.yaml`; `.env` and
shell environment overrides take precedence. Supported maximums are 3,600 seconds
for provider requests, routing, and steps, and 14,400 seconds for a whole task.
Longer timeouts allow more waiting; they do not speed up inference. Stop still
interrupts a running task.

Restart the backend with Ctrl+C, then run from the project root:

```powershell
.\scripts\start-backend.ps1
```

## Development and release checks

```powershell
.\scripts\build.ps1
.\scripts\test.ps1 -Fast
.\scripts\doctor.ps1
```

For the full suite, including real Chrome tests on synthetic local fixtures:

```powershell
.\scripts\test.ps1
```

Browser tests use synthetic local pages and an isolated temporary Chrome profile;
they do not use your personal profile. They require an interactive Windows desktop.
Skipped checks are reported as incomplete. Passing fixture tests is not live
website/model acceptance.

- [Architecture](docs/architecture.md)
- [Data storage and privacy](docs/data-storage.md)
- [Testing and development](docs/testing.md)
- [Configuration example](backend/.env.example)
- [Pinned backend dependencies](backend/requirements.txt)

## Chrome permissions

No required `<all_urls>`, `tabs`, `tabGroups`, cookies, debugger or webRequest permission.
No content script runs automatically on every page.

| Permission | Why it remains |
| --- | --- |
| `activeTab` | Clicking the toolbar authorizes temporary access to the current tab. |
| `scripting` | Inject and call the real-DOM observation/action runtime in that authorized tab. |
| `sidePanel` | Display the assistant in a separate, tab-bound native panel. |
| `storage` | Store the local backend connection settings and token; never Chrome sync. |
| `downloads` | Observe and validate explicit file downloads, enforce size/status checks, and prepare selected PDFs locally. |
| `declarativeNetRequest` | Temporary exact-URL rules prevent selected file acquisition from following a different main-frame target; rules are removed when the acquisition lease ends. |
| Required hosts `http://127.0.0.1/*`, `http://localhost/*` | Reach the local backend health endpoint. Chrome match patterns cannot restrict a port. |
| Optional hosts `http://*/*`, `https://*/*` | Permit user-selected persistent site grants. These are not granted at installation. |

Chrome's [activeTab documentation](https://developer.chrome.com/docs/extensions/develop/concepts/activeTab)
explains origin changes and restricted pages. Clicking a link, typing, searching,
scrolling, navigation, back/forward and refresh have no agent confirmation modal.
Only recognized deletion/removal/destructive actions require one confirmation for that action.
A remote PDF-context consent control remains explicit when choosing a cloud model; it
is a data-sharing choice, not a browser-action confirmation.

## Providers and private data

Ollama/local, OpenAI, Azure OpenAI, Anthropic, compatible endpoints and mock remain
available in `backend/config.example.yaml`. The small local router selects a capability;
the configurable main model decides actions and answers. Provider API keys never go
into extension storage. Page data sent to a selected cloud provider leaves this computer.

PDF preparation preserves exact observed URL/file validation, byte hashing, extraction
limits, local BM25/MiniLM and optional Nomic multimodal retrieval. Generic file cards
are derived from visible links; application-specific preview URLs and extensionless
file endpoints are not automatically interpreted as PDFs. Local CLI ingestion remains
available with `tools/pdf_ingest.py`, `tools/local_retrieval.py` and `tools/local_rag.py`.

The neutral private data root is `AIBrowserAgent` on Windows (or `ai-browser-agent`
on other platforms); a local checkout can use ignored `.local/`. Existing private
stores are not moved automatically. Keep explicit store/model path overrides if needed.

Before publication run `python tools/public_privacy_scan.py` and review its path-only
findings. It never prints matched secret values. Keep the ignored `archive/` directory out of public distributions. Review Git history separately; deleting working-tree files does not remove history.
Never publish `.env`, downloads, private PDFs, audit logs, screenshots, databases, browser
profiles or model caches. The audit is a pattern scan, not a guarantee that no secret exists.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
Third-party dependencies and model weights remain subject to their respective licenses.
