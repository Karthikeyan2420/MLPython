# import spacy

# nlp = spacy.load("en_core_web_sm")

# text = "Karthi Raja works at Infosys in Chennai since 2022."
# doc = nlp(text)

# for ent in doc.ents:
#     print(ent.text, "->", ent.label_)


from transformers import pipeline

nlp = pipeline("ner")
text = "Invoice No: 4567 Date: 12 Aug 2024 Amount: Rs 5000"

result = nlp(text)
print(result)

