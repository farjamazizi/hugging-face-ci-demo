# from transformers import pipeline
# import gradio as gr


# model = pipeline(
#     "summarization",
# )

# def predict(prompt):
#     summary = model(prompt)[0]["summary_text"]
#     return summary


# # create an interface for the model
# with gr.Interface(predict, "textbox", "text") as interface:
#     interface.launch()
    
from transformers import pipeline
import gradio as gr

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",   # smaller & faster than bart-large-cnn
    framework="pt"                           # force PyTorch backend
)

def predict(text: str) -> str:
    if not text or not text.strip():
        return "Please enter some text to summarize."
    out = summarizer(text, max_length=130, min_length=30, do_sample=False, truncation=True)
    return out[0]["summary_text"]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=6, placeholder="Paste text to summarize..."),
    outputs=gr.Textbox(label="Summary"),
    title="First Demo — Text Summarizer",
)

if __name__ == "__main__":
    demo.launch()
    