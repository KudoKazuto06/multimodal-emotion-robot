
# Real-Time Multimodal Emotion Recognition for Interactive Robots

A lightweight, local multimodal machine learning prototype that combines **text and speech audio** to recognize emotions and generate short, context-aware conversational responses.

Built using the **MELD (Multimodal EmotionLines Dataset)** for the Real-Time Multimodal ML Challenge.

The system accepts a spoken utterance and its transcript, predicts one of seven emotion categories, and generates a short response appropriate to the predicted emotion. It also provides a browser-based interface for repeated interactions.

## 1. Project Overview

Emotion-aware interactions require more than understanding the literal meaning of words. Speech characteristics such as vocal intensity, pitch-related information, and speaking patterns may provide additional clues about the speaker's emotional state.

This project investigates whether combining textual information with acoustic features can improve emotion classification while maintaining low computational requirements.

The prototype prioritizes:

- A complete, working text-and-audio inference pipeline.
- Local inference without an external API.
- Fast response generation suitable for utterance-level interaction.
- Reproducible training and evaluation.
- A lightweight architecture that runs on a CPU.
- Transparent reporting of performance and limitations.

### Supported Emotions

The classifier predicts one of seven MELD categories:

- Anger
- Disgust
- Fear
- Joy
- Neutral
- Sadness
- Surprise

### Input and Output

**Input**

- Utterance transcript (text).
- Corresponding speech audio (uploaded or microphone-recorded).

**Output**

```json
{
  "emotion": "surprise",
  "confidence": 0.2879,
  "scores": {
    "anger": 0.0472,
    "disgust": 0.1411,
    "fear": 0.0874,
    "joy": 0.1652,
    "neutral": 0.0723,
    "sadness": 0.199,
    "surprise": 0.2879
  },
  "response": "That sounds unexpected! What happened next?",
  "latency_ms": 1998.03
}
```

This example is a recorded first-call result, including initialization overhead. Warmed-up inference is substantially faster.

The confidence values represent classifier probability estimates and should not be interpreted as calibrated confidence.

## 2. System Architecture

The architecture uses feature-level fusion between textual and acoustic representations.

```text
           Input Utterance
                  |
         +--------+--------+
         |                 |
       Text               Audio
         |                 |
   TF-IDF Vectorizer    Resample to 16 kHz
         |                 |
   20,000 Features      20 MFCC Coefficients
         |                 |
         |              Mean + Std
         |                 |
         |             40 Features
         |                 |
         |           StandardScaler
         |                 |
         +--------+--------+
                  |
           Feature Concatenation
                  |
          Logistic Regression
                  |
          7 Emotion Categories
                  |
          Rule-Based Response
                  |
           Structured Output
                  |
             Gradio Demo
```

### 2.1 Text Feature Extraction

The text branch uses scikit-learn's `TfidfVectorizer`.

Configuration:

| Parameter | Value |
|---|---|
| Maximum vocabulary size | 20,000 |
| N-gram range | (1, 2) |
| Sublinear term frequency | Enabled |
| Representation | Sparse TF-IDF vector |

Unigrams and bigrams capture individual words and short expressions.

TF-IDF provides a lightweight representation without requiring transformer inference.

### 2.2 Audio Feature Extraction

Audio is processed using Librosa.

Processing steps:

1. Load the audio recording.
2. Convert to mono.
3. Resample to 16,000 Hz.
4. Extract 20 Mel-frequency cepstral coefficients (MFCCs).
5. Compute the mean and standard deviation of each coefficient across frames.
6. Concatenate the statistics into a 40-dimensional vector.
7. Standardize the features using a scaler fitted on the training set.

The resulting feature vector is:

```text
[MFCC_1_mean, ..., MFCC_20_mean,
 MFCC_1_std,  ..., MFCC_20_std]
```

MFCCs summarize aspects of the speech spectrum. They are computationally efficient, although they do not retain all temporal or prosodic information.

### 2.3 Multimodal Fusion

The text and standardized audio features are concatenated:

```text
Combined Representation = [TF-IDF Features | Audio Features]
```

The final feature dimension is:

```text
20,000 + 40 = 20,040
```

The classifier is a multinomial Logistic Regression model using scikit-learn.

Training configuration:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs"
)
```

Balanced class weighting is used to address the uneven emotion distribution in MELD.

### 2.4 Response Generation

A lightweight rule-based module generates a response from the predicted emotion and utterance text.

For example:

| Emotion | Example response |
|---|---|
| Anger | That sounds frustrating. Want to tell me what happened? |
| Fear | That sounds worrying. Would you like to talk it through? |
| Joy | That's wonderful! What made it so exciting? |
| Sadness | I'm sorry that's difficult. Would you like to talk about it? |
| Surprise | That sounds unexpected! What happened next? |
| Neutral | I'm listening. Tell me more. |
| Disgust | That sounds unpleasant. What bothered you about it? |

The response generator does not use a large language model. This keeps the entire system lightweight and deterministic but limits conversational flexibility.

## 3. Dataset

The project uses the MELD dataset, which contains emotion-labeled utterances from multi-party conversations.

Dataset splits:

| Split | Original utterances | Usable examples |
|---|---:|---:|
| Train | 9,989 | 9,988 |
| Development | 1,109 | 1,108 |
| Test | 2,610 | 2,610 |
| **Total** | **13,708** | **13,706** |

Two utterances were excluded because their corresponding audio was unavailable or invalid.

### Preprocessing

The preprocessing pipeline:

1. Reads the MELD metadata CSV files.
2. Identifies utterance audio recordings using dialogue and utterance IDs.
3. Converts the source video audio tracks to mono, 16 kHz WAV files.
4. Validates the alignment between text, labels, and audio.
5. Removes examples without usable audio.
6. Extracts 40-dimensional MFCC summary features.
7. Saves aligned manifests and feature arrays.

The test split is held out from model fitting.

### Dataset Availability

The MELD dataset is not redistributed in this repository.

Users should obtain it from the official dataset source:

https://github.com/declare-lab/MELD

The training and evaluation scripts expect the preprocessed manifests and feature arrays generated by the data preparation pipeline.

## 4. Project Structure

```text
multimodel-emotion-robot/
|
|-- app.py
|-- README.md
|-- requirements.txt
|
|-- src/
|   |-- __init__.py
|   |-- inference.py
|   |-- response.py
|
|-- scripts/
|   |-- verify_dataset.py
|   |-- prepare_data.py
|   |-- extract_audio_features.py
|   |-- train_text_baseline.py
|   |-- train_multimodal.py
|   |-- test_inference.py
|   |-- evaluate_final.py
|   |-- benchmark.py
|   |-- profile_model.py
|
|-- checkpoints/
|   |-- text_baseline.joblib
|   |-- multimodal_model.joblib
|
|-- results/
|   |-- final_test_metrics.json
|   |-- confusion_matrix.png
|   |-- latency_benchmark.json
|   |-- resource_profile.json
|
|-- data/
|   |-- raw/
|   |-- processed/
|
|-- configs/
|-- tests/
```

The large raw MELD files and extracted audio recordings are excluded from version control.

## 5. Installation

### Requirements

- Python 3.13.5 (tested)
- Windows (tested)
- Approximately 200 MB of observed process memory after inference warm-up, excluding other applications
- CPU inference supported
- No CUDA-capable GPU required for deployment

### Clone Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd multimodel-emotion-robot
```

### Create Virtual Environment

On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

Primary runtime libraries:

- scikit-learn
- NumPy
- SciPy
- Librosa
- SoundFile
- Joblib
- Pandas
- Gradio

Evaluation utilities additionally use Matplotlib, tqdm, and psutil.

A CUDA-enabled PyTorch installation is not required for this model's inference pipeline.

**Note:** Serialized Joblib checkpoints should only be loaded from trusted sources.

## 6. Running the Application

Launch the interactive Gradio interface:

```powershell
python app.py
```

Open the local URL displayed in the terminal, typically:

```text
http://127.0.0.1:7860
```

The interface supports:

- Entering an utterance transcript.
- Uploading an audio recording.
- Recording audio through the microphone.
- Predicting one of seven emotion categories.
- Displaying model scores.
- Generating a short response.
- Displaying measured inference latency.
- Maintaining a session-level interaction history.

For microphone input, the transcript must be entered manually. Automatic speech recognition is not implemented.

### Command-Line Inference

Test a development-set example:

```powershell
python -m scripts.test_inference
```

This command requires the preprocessed development manifest and corresponding audio.

## 7. Experimental Evaluation

Two classifiers are evaluated:

1. **Text-only baseline:** TF-IDF + Logistic Regression.
2. **Multimodal model:** TF-IDF + MFCC + Logistic Regression.

Both models are evaluated on the same held-out test split containing 2,610 utterances.

### 7.1 Final Test Results

| Metric | Text Only | Text + Audio | Improvement |
|---|---:|---:|---:|
| Accuracy | 40.54% | 42.26% | +1.72 percentage points |
| Macro-F1 | 0.3142 | 0.3266 | +0.0124 |
| Weighted-F1 | 0.4248 | 0.4430 | +0.0183 |

Adding acoustic features produces a modest improvement across all three aggregate metrics.

### 7.2 Per-Class F1

| Emotion | Text Only | Text + Audio |
|---|---:|---:|
| Anger | 0.29 | 0.37 |
| Disgust | 0.16 | 0.16 |
| Fear | 0.08 | 0.09 |
| Joy | 0.41 | 0.39 |
| Neutral | 0.51 | 0.53 |
| Sadness | 0.29 | 0.28 |
| Surprise | 0.46 | 0.47 |

The largest observed improvement occurs for anger. Other categories show smaller or mixed changes.

### 7.3 Class Imbalance

The test set contains 1,256 neutral utterances out of 2,610 examples.

An always-neutral classifier would achieve approximately 48.12% accuracy but a macro-F1 of approximately 0.093.

Therefore, overall accuracy alone is not sufficient to evaluate this task. Macro-F1 is included to capture performance across minority emotion categories.

The multimodal model improves upon the text-only baseline but still has limited classification performance, especially for fear and disgust.

### 7.4 Reproducing Evaluation

After preparing the MELD test split:

```powershell
python -m scripts.evaluate_final
```

This generates:

```text
results/final_test_metrics.json
results/confusion_matrix.png
```

No parameters are updated during evaluation.

## 8. Real-Time Inference

### Definition of Real-Time

This prototype defines near-real-time interaction as producing an emotion prediction and response shortly after a complete utterance and its transcript become available.

It is an **utterance-level** system rather than a continuous streaming recognizer.

The target local processing latency is 100 milliseconds per utterance.

### 8.1 Latency Benchmark

The benchmark processes 100 development-set utterances after five warm-up runs.

Measured results:

| Metric | Latency |
|---|---:|
| Mean | 6.49 ms |
| Median | 5.63 ms |
| 95th percentile | 10.32 ms |
| Minimum | 3.68 ms |
| Maximum | 39.87 ms |
| Within 100 ms | 100% of tested calls |

The measurements include:

- Reading the audio file.
- MFCC extraction.
- TF-IDF transformation.
- Audio feature scaling.
- Emotion classification.
- Rule-based response generation.

They exclude:

- Initial model loading.
- Microphone recording.
- Time spent speaking.
- Manual transcript entry.
- Browser interface rendering.

Consequently, the benchmark demonstrates low local processing latency, not complete speech-to-response latency.

### Reproduce Benchmark

```powershell
python -m scripts.benchmark
```

Output:

```text
results/latency_benchmark.json
```

## 9. Model Size and Hardware Requirements

### 9.1 Model Parameters

| Component | Count |
|---|---:|
| Logistic Regression coefficients and intercepts | 140,287 |
| Fitted TF-IDF IDF statistics | 20,000 |
| Stored audio scaler statistics | 120 |
| Total stored fitted numerical values | 160,407 |

The classifier contains **140,287 learned coefficients and intercepts**.

The total of 160,407 additionally includes stored preprocessing statistics and is reported separately from classifier parameters.

The model does not use pretrained transformer, speech encoder, or generative language model weights.

**6-billion-parameter constraint: satisfied.**

### 9.2 Model Storage

| Property | Value |
|---|---|
| Multimodal checkpoint | 1.812 MiB |
| Runtime inference device | CPU |
| GPU required | No |

### 9.3 Measured Hardware

Development and evaluation were conducted on:

| Component | Specification |
|---|---|
| CPU | Intel processor, 12 logical cores |
| System RAM | 15.69 GiB |
| Available GPU | NVIDIA GeForce MX550, 2 GiB |
| Python | 3.13.5 |
| Inference | CPU |

### 9.4 Observed Memory Usage

| Measurement | Process RAM |
|---|---:|
| Before loading model checkpoint | 153.41 MiB |
| After loading model checkpoint | 156.77 MiB |
| After first inference warm-up | 194.68 MiB |

These are snapshots of process resident memory, not measurements of peak memory usage or minimum system requirements.

The model operates without CUDA acceleration.

### Reproduce Resource Profile

```powershell
python -m scripts.profile_model
```

Output:

```text
results/resource_profile.json
```

## 10. Design Decisions and Trade-offs

### Why Logistic Regression?

The challenge prioritizes a complete, understandable prototype rather than an unfinished large model.

Logistic Regression was selected because it:

- Supports fast CPU inference.
- Works efficiently with sparse TF-IDF vectors.
- Is straightforward to train and evaluate.
- Requires relatively little memory.
- Provides class probability estimates.

The approach offers a simple baseline for investigating whether audio contributes useful information beyond text.

### Why MFCC Features?

MFCCs are widely used acoustic features that provide a compact representation of the speech spectrum.

Summarizing them using mean and standard deviation provides a fixed-size representation for utterances of different durations.

However, this summary removes temporal information and may miss changes in speaking style across an utterance.

### Why Rule-Based Responses?

Using predefined responses eliminates the latency, memory requirements, and additional dependencies of a generative language model.

It also produces predictable output.

The limitation is that replies are generic and depend heavily on the emotion classification being correct.

## 11. Limitations

The current prototype has several limitations.

**Classification performance:** The model achieves 42.26% accuracy and 0.3266 macro-F1 on the held-out MELD test set. Some emotion categories remain difficult to distinguish.

**Class imbalance:** Neutral is the largest category, while fear and disgust contain relatively few examples.

**Audio representation:** Averaged MFCC statistics do not capture the full temporal dynamics of speech. Very short audio clips may also produce Librosa FFT-size warnings.

**No conversational context:** Each utterance is classified independently. Previous utterances displayed in the Gradio history are not fed back into the classifier.

**No automatic transcription:** The interface requires a matching text transcript alongside the audio recording.

**No continuous streaming:** Predictions occur after an entire utterance is provided.

**Rule-based response generation:** The output is emotion-conditioned but not generated by a large language model, limiting flexibility and contextual depth.

**Domain generalization:** MELD consists of scripted television dialogue. Performance on spontaneous speech, unfamiliar microphones, accents, noisy environments, or real robotic interactions has not been established.

**Probability calibration:** Model scores have not been calibrated and should not be treated as reliable certainty estimates.

## 12. Possible Future Improvements

Future work could investigate:

- Pretrained speech embeddings that capture more acoustic and prosodic information.
- Contextual text representations.
- Dialogue-history-aware emotion modeling.
- More sophisticated multimodal fusion.
- Confidence calibration and abstention for uncertain predictions.
- Automatic speech recognition.
- Streaming audio and incremental inference.
- More context-sensitive response generation.

These extensions are outside the scope of the current lightweight prototype.

## 13. AI-Assisted Development Disclosure

ChatGPT was used as a development assistant during this project.

Assistance included:

- Breaking down the challenge into implementation stages.
- Explaining machine learning concepts and design options.
- Suggesting project structure and workflow.
- Drafting and debugging portions of Python code.
- Developing inference, benchmarking, evaluation, and demonstration scripts.
- Assisting with technical documentation.

The implementation was executed, debugged, and evaluated locally. Reported performance, latency, and resource measurements come from actual local experiments rather than AI-generated estimates.

The final classifier is a scikit-learn model trained using the MELD training data. ChatGPT is not part of the deployed inference pipeline.

## 14. Summary

This project delivers an end-to-end, lightweight emotion-aware interaction prototype using text and audio.

Key results:

- Seven-class emotion classification.
- Feature-level fusion of 20,000 text features and 40 audio features.
- Held-out test accuracy of 42.26%.
- Held-out macro-F1 of 0.3266.
- Measured mean local inference latency of 6.49 ms.
- Measured P95 local inference latency of 10.32 ms.
- 140,287 classifier parameters.
- 1.812 MiB multimodal checkpoint.
- Local CPU execution without CUDA.
- Interactive Gradio demonstration with repeated utterance-level interactions.

The main contribution is a reproducible, low-resource end-to-end prototype that demonstrates both the benefits and limitations of combining textual and acoustic information for emotion-aware interaction.
