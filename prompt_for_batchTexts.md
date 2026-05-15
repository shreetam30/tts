I have a Streamlit TTS benchmarking app. I want to add a Batch Audio Generation feature inside the existing Generate tab (tab1) as a second mode, toggled by a radio button. Here is the exact spec:
UI Structure inside tab1:

Radio toggle at top: ["🎙 Single Generate", "🗃 Batch Generate"]
When "🗃 Batch Generate" is selected, show the batch UI below

Batch UI — Step 1: Load Sentences

Radio: ["📄 Upload CSV", "✏️ Paste Text"]
CSV upload: read with pd.read_csv(uploaded_csv, header=None, dtype=str), let user pick column if multiple columns exist, strip leading bullets/asterisks/numbering with re.sub(r"^[\*\-\•\d]+[\.\):\s]+", "", s).strip()
Paste text: same stripping logic, one sentence per line
Critical: After parsing, immediately store sentences in st.session_state["_batch_sentences_ready"] = cleaned_list. This is mandatory because Streamlit reruns on button click and local variables are lost.
Show parsed count and preview (first 20 sentences)

Batch UI — Step 2: Select Model

Selectbox from filtered dict (already available in scope), key="batch_model_select"
Show model info card

Batch UI — Step 3: Options

Three checkboxes: "Compute WER/CER metrics" (key="batch_compute_wer"), "Skip errors & continue" (key="batch_skip_errors"), "Save results JSON" (key="batch_save_json")

Batch UI — Step 4: Generate button

Read sentences from st.session_state.get("_batch_sentences_ready", []) — NOT from local variable
Disable button if not _ready_sentences or not _batch_files_ok
Button label: f"▶ Generate {len(_ready_sentences)} Audio File(s)"
On click: batch_sentences = _ready_sentences then loop

Generation loop:
pythonBATCH_OUTPUT_DIR = BASE_DIR / "batch_outputs"
BATCH_OUTPUT_DIR.mkdir(exist_ok=True)

for i, sentence in enumerate(batch_sentences, start=1):
    det_lang = detect_language(sentence)
    cfg = MODEL_REGISTRY[batch_model]
    if det_lang not in cfg.get("langs", ["en"]):
        det_lang = cfg.get("langs", ["en"])[0]
    
    # filename format: {index}_{model_slug}_{lang}.wav
    model_slug = batch_model.lower().replace(" ","_").replace("/","_").replace("(","").replace(")","")[:30]
    out_filename = f"{i}_{model_slug}_{det_lang}.wav"
    out_path = BATCH_OUTPUT_DIR / out_filename
    
    # call existing generate() function
    (wav_path, audio, sr, gen_time, duration, rtf, cpu_pct, mem_mb, det_lang_out) = generate(batch_model, sentence)
    shutil.copy2(wav_path, str(out_path))
    
    # optionally compute metrics using existing compute_metrics()
    # append result dict with: index, input_text, lang, wav_path, filename, status, error, metrics
    
    # update live progress table every 5 rows
Results section (shown after generation and persisted via session state):

Store in st.session_state["batch_results_df"] and st.session_state["batch_results_list"]
Show summary metrics (avg RTF, WER, etc.) using st.metric
Show full results dataframe
Audio playback section with expanders per sentence, toggled by a checkbox
Download buttons: Results CSV, Results JSON, ZIP of all WAV files
Clear button that clears batch_results_df, batch_results_list, _batch_sentences_ready from session state and calls st.rerun()

Critical constraints:

All widget keys in batch mode must be unique and different from any keys used in Single Generate mode or any other tab. Prefix all batch keys with "batch_".
Do NOT create a separate tab9. Everything goes inside with tab1: under elif gen_mode == "🗃 Batch Generate":.
The generate button MUST read sentences from st.session_state["_batch_sentences_ready"], not from a local variable — this is the core fix for why the button appears disabled or does nothing after clicking.
filtered is a dict of {model_name: cfg} already available from sidebar scope.
These functions already exist and must be reused: generate(model_name, text), compute_metrics(wav_path, ref_text, gen_time, duration, rtf, cpu_pct, mem_mb), detect_language(text), get_model_file_status(model_name), generate_batch_filename(index, model_name, lang, text).
MODEL_REGISTRY, BASE_DIR, LANG_LABEL, BATCH_OUTPUT_DIR are all available in scope.
The tab declaration must be exactly 8 tabs: tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([...]) — no tab9.

Existing generate() return signature:
pythondef generate(model_name: str, text: str):
    # returns: wav_path, audio, sr, gen_time, duration, rtf, cpu_pct, mem_mb, det_lang
Existing compute_metrics() signature:
pythondef compute_metrics(wav_path, ref_text, gen_time, duration, rtf, cpu_pct, mem_mb):
    # returns dict with: asr_transcript, wer, cer, mos_proxy, generation_time_s,
    # audio_duration_s, rtf, cpu_model_pct, memory_mb
Please give me only the complete with tab1: block, ready to paste, with no explanation outside the code block.
