import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

document = "Natural language processing (NLP) is a field of computer science, artificial intelligence, and computational linguistics concerned with the interactions between computers and human (natural) languages."

tokens = word_tokenize(document)

pos_tags = pos_tag(tokens)

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

print("Original Document:")
print(document)
print("\nTokenization:")
print(tokens)
print("\nPOS Tagging:")
print(pos_tags)
print("\nStopwords Removal:")
print(filtered_tokens)
print("\nStemming:")
print(stemmed_tokens)
print("\nLemmatization:")
print(lemmatized_tokens)


from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    
]

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("TF-IDF Representation:")
print(tfidf_matrix.toarray())

