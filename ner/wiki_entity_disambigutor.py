from ner.SimplifiedLesk import SimplifiedLesk
from ner.sentence_similarity import cal_sentence_similarity
from nltk.corpus import stopwords
from mediawiki import MediaWiki, exceptions
from joblib import Parallel, delayed

# Defining stopwords for English Language
stop_words = set(stopwords.words("english"))


# Extract POS from the doc
def extract_pos_frm_doc(spacy_doc, input_sentence=None, entities=None):
    token_dict = {}
    simplified_lesk = SimplifiedLesk()

    for token in spacy_doc:
        if token.text not in stop_words \
                and not token.is_punct \
                and not token.is_space \
                and not token.is_bracket \
                and not token.is_quote \
                and token.pos_ not in ['DET', 'PRON', 'PART'] \
                and (entities is None or token.text not in entities) \
                and token.ent_type_ is not 'DATE':
            if token.pos_ in token_dict.keys():
                if input_sentence is None:
                    token_dict[token.pos_] = token_dict[token.pos_] + " " + token.text
                else:
                    word_sens = simplified_lesk.disambiguate(token.text, input_sentence)
                    if word_sens is not None:
                        key_word_def = word_sens.definition()
                        token_dict[token.pos_] = token_dict[token.pos_] + " " + key_word_def
            else:
                if input_sentence is None:
                    token_dict[token.pos_] = token.text
                else:
                    word_sense = simplified_lesk.disambiguate(token.text, input_sentence)
                    if word_sense is not None:
                        token_dict[token.pos_] = word_sense.definition()
                        # token_dict[token.pos_] = word_sense
    return token_dict


def cal_similarity_index_wiki_page(wiki_page_title, sent_token_pos_dict, nlp):
    final_similarity_index = 0
    # wiki page creation and summary
    wikipedia = MediaWiki()
    wiki_page = wikipedia.page(wiki_page_title, auto_suggest=False, redirect=False)
    wiki_page_summary = wiki_page.summary

    # wiki page summary keyword dict
    wiki_page_summary_pos_dict = extract_pos_frm_doc(nlp(wiki_page_summary))

    for key, value in wiki_page_summary_pos_dict.items():
        if key in sent_token_pos_dict.keys():
            sent = sent_token_pos_dict[key]
            sentence_sim = cal_sentence_similarity(value, sent)

            if sentence_sim > final_similarity_index:
                final_similarity_index = sentence_sim

    return final_similarity_index, wiki_page.pageid
