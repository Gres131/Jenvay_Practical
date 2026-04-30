import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# 1. LOAD & LIMIT DATA
# =========================
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

text = text[:250000]  # 🔥 balanced size

# =========================
# 2. TOKENIZATION (LIMIT VOCAB)
# =========================
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts([text])

total_words = min(5000, len(tokenizer.word_index)) + 1

# =========================
# 3. SEQUENCE CREATION (EFFICIENT)
# =========================
max_seq_len = 25  # 🔥 small = fast + low RAM

sequences = []

for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        seq = token_list[max(0, i - max_seq_len):i+1]
        sequences.append(seq)

sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='pre')

X = sequences[:, :-1]
y = sequences[:, -1]

# =========================
# 4. MODEL (OPTIMIZED)
# =========================
model = Sequential([
    Embedding(total_words, 64, input_length=max_seq_len-1),

    LSTM(96, return_sequences=True),
    Dropout(0.2),

    LSTM(48),
    Dropout(0.2),

    Dense(total_words, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

model.summary()

# =========================
# 5. TRAINING (SMART)
# =========================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=3,
    restore_best_weights=True
)

model.fit(X, y, epochs=20, batch_size=64, callbacks=[early_stop])

# =========================
# 6. TEMPERATURE SAMPLING
# =========================
def sample_with_temperature(preds, temperature=0.8):
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# =========================
# 7. TEXT GENERATION
# =========================
def generate_text(seed_text, next_words=20, temperature=0.8):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

        preds = model.predict(token_list, verbose=0)[0]
        predicted_index = sample_with_temperature(preds, temperature)

        word = ""
        for w, idx in tokenizer.word_index.items():
            if idx == predicted_index:
                word = w
                break

        seed_text += " " + word

    return seed_text

# =========================
# 8. GENERATE RESULTS
# =========================
result1 = generate_text("king", 20, temperature=0.7)
result2 = generate_text("love", 20, temperature=0.8)

print("\nGenerated Text:\n")
print(result1)
print(result2)

# =========================
# 9. SAVE TO PDF
# =========================
output_dir = "Result"
os.makedirs(output_dir, exist_ok=True)

pdf_path = os.path.join(output_dir, "generated_text_results.pdf")

doc = SimpleDocTemplate(pdf_path)
styles = getSampleStyleSheet()

content = []

content.append(Paragraph("<b>Optimized Text Generation Results</b>", styles["Title"]))
content.append(Spacer(1, 20))

content.append(Paragraph("<b>Seed: king</b>", styles["Heading2"]))
content.append(Paragraph(result1, styles["BodyText"]))
content.append(Spacer(1, 15))

content.append(Paragraph("<b>Seed: love</b>", styles["Heading2"]))
content.append(Paragraph(result2, styles["BodyText"]))

doc.build(content)

print(f"\n✅ PDF saved at: {pdf_path}")
