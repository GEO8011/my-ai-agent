from faster_whisper import WhisperModel
print("Φορτώνω το μοντέλο tiny... (πρώτη φορά κατεβάζει ~75 MB)")
model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("✅ Το μοντέλο φορτώθηκε!")

# Δοκιμή με ένα μικρό αρχείο (αν έχεις)
# segments, info = model.transcribe("test.wav", language="el")
# for seg in segments:
#     print(seg.text)