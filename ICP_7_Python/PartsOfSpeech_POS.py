import nltk

nltk.download('averaged_perceptron_tagger')
f = open('sravan.txt', 'r', encoding='utf-8')
input = f.read()


wtokens = nltk.word_tokenize(input)

print(nltk.pos_tag(wtokens))
