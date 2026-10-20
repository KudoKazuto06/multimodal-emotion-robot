
"""Interactive Text + Audio emotion recognition demo."""

from datetime import datetime

import gradio as gr

from src.inference import EmotionPredictor


# Load model once when the application starts
predictor = EmotionPredictor()


def predict_emotion(text, audio, history):
    """
    Process one utterance and return:
    - Predicted emotion
    - Model confidence
    - Robot response
    - Emotion scores
    - Inference latency
    - Updated interaction history
    """

    if not text or not text.strip():
        raise gr.Error("Please enter the spoken text.")

    if audio is None:
        raise gr.Error("Please upload or record audio.")

    result = predictor.predict(
        text=text.strip(),
        audio_path=audio
    )

    emotion = result["emotion"]
    confidence = result["confidence"]
    response = result["response"]
    latency = result["latency_ms"]

    # Display the emotion and confidence
    emotion_result = (
        f"### {emotion.capitalize()}\n"
        f"Model score: {confidence:.2%}"
    )

    # Format processing latency
    latency_result = f"{latency:.2f} ms"

    # Maintain per-session interaction history
    history = list(history or [])

    history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "text": text.strip(),
        "emotion": emotion,
        "confidence": round(confidence, 4),
        "response": response,
        "latency_ms": latency,
    })

    # Display recent interactions
    history_lines = []

    for index, item in enumerate(history, start=1):
        history_lines.append(
            f"**Interaction {index} — {item['time']}**\n\n"
            f"**User:** {item['text']}\n\n"
            f"**Emotion:** {item['emotion'].capitalize()} "
            f"({item['confidence']:.1%})\n\n"
            f"**Robot:** {item['response']}\n\n"
            f"---"
        )

    history_text = "\n\n".join(history_lines)

    return (
        emotion_result,
        response,
        result["scores"],
        latency_result,
        history_text,
        history,
    )


def clear_history():
    return "", []


with gr.Blocks(
    title="Multimodal Emotion Robot"
) as demo:

    gr.Markdown(
        """
        # Multimodal Emotion Robot

        ### Real-Time Text + Audio Emotion Recognition

        This prototype combines utterance text and speech
        audio to predict one of seven MELD emotion categories
        and generate a short conversational response.

        **Model:** TF-IDF + MFCC + Logistic Regression

        **Inference:** Local, utterance-level processing
        """
    )

    history_state = gr.State([])

    with gr.Row():

        with gr.Column():

            gr.Markdown("## Inputs")

            text_input = gr.Textbox(
                label="Utterance text",
                placeholder="Type what the speaker said...",
                lines=3
            )

            audio_input = gr.Audio(
                label="Speech audio",
                sources=["upload", "microphone"],
                type="filepath",
                streaming=False
            )

            predict_button = gr.Button(
                "Analyze Emotion",
                variant="primary"
            )

        with gr.Column():

            gr.Markdown("## Prediction")

            emotion_output = gr.Markdown(
                value="### No prediction yet"
            )

            response_output = gr.Textbox(
                label="Robot response",
                interactive=False,
                lines=3
            )

            scores_output = gr.Label(
                label="Emotion scores",
                num_top_classes=7
            )

            latency_output = gr.Textbox(
                label="Inference latency",
                interactive=False
            )

    gr.Markdown("## Interaction History")

    history_output = gr.Markdown(
        value="No interactions yet."
    )

    clear_button = gr.Button(
        "Clear History"
    )

    predict_button.click(
        fn=predict_emotion,
        inputs=[
            text_input,
            audio_input,
            history_state
        ],
        outputs=[
            emotion_output,
            response_output,
            scores_output,
            latency_output,
            history_output,
            history_state
        ]
    )

    clear_button.click(
        fn=clear_history,
        inputs=[],
        outputs=[
            history_output,
            history_state
        ]
    )

    gr.Markdown(
        """
        ---
        **Important:** Model scores are uncalibrated estimates.
        The model processes complete utterances, not a continuous
        audio stream. Interaction history is displayed but is
        not used by the emotion classifier.
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
