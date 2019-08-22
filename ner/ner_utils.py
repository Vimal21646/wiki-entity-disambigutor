from spacy.lang.en import English


# Method to split the text into sentences
def split_string_sentences(raw_text):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))  # updated
    doc = nlp(raw_text)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences


def to_string_lexeme_list(similarity):
    similar_words = ","
    # Map the lexeme into string list
    similarity_words = map(lambda word: word.text, similarity[:10])
    # Join the the strings in the list
    similar_words = similar_words.join(similarity_words)
    return similar_words


# Method to find the given sub strig index in the given string
def find_in_str(src_string, srch_text='.'):
    # variable initialization
    index_srch_text = src_string.find(srch_text)
    result_str = ''

    # get the string
    if index_srch_text != -1:
        result_str = src_string[:index_srch_text + 1]

    return result_str


# Method to get the substring from the given character
def find_sub_string_till_end(src_string, srch_string="/"):
    start_index = src_string.rfind(srch_string) + 1
    sub_str = src_string[start_index:]
    return sub_str

