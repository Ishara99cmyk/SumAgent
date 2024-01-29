from flask import Flask, render_template, request
from transformers import pipeline,T5ForConditionalGeneration,T5Tokenizer

app = Flask(__name__)

# Load the pre-trained model and tokenizer
trained_model = T5ForConditionalGeneration.from_pretrained("personalized_summarizer_model")
saved_tokenizer = T5Tokenizer.from_pretrained("personalized_tokenizer")
summarizer = pipeline("summarization", model=trained_model, tokenizer=saved_tokenizer)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        user_input = request.form['user_input']
        # Generate summary using the loaded model
        summary = summarizer(user_input, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, no_repeat_ngram_size=2)
        return render_template('index.html', user_input=user_input, summary=summary[0]['summary_text'])

if __name__ == '__main__':
    app.run(debug=True)

