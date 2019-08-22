from flask import Flask, request, jsonify
import logging
import ner.ner_constant as nc
import ner.spacy_pipelines as spacy_pipe
import spacy
import ner.wiki_entity_disambigutor as wiki_dis

# Initialize the Flask application
app = Flask(__name__)


# Service to extract entities using text
@app.route('/api/get_wiki_page_sim_index', methods=['POST'])
def get_wiki_page_sim_index():
    # Get the json data from request
    content = request.json
    wiki_page_title = content['wiki_page_title']
    sent_token_pos_dict = content["sent_token_pos_dict"]
    nlp = spacy.load(nc.SPACY_MODEL)
    nlp.add_pipe(spacy_pipe.expand_person_entities, after='ner')

    final_similarity_index, wiki_page_id = wiki_dis.cal_similarity_index_wiki_page(wiki_page_title, sent_token_pos_dict,
                                                                                   nlp)
    return_dict = {"wiki_page": wiki_page_id, "final_similarity_index": final_similarity_index}
    return jsonify(return_dict)


@app.route('/api/get_test_html', methods=['GET'])
def get_test_html_text():
    final_similarity_index = 1
    from mediawiki import MediaWiki
    wikipedia = MediaWiki()
    wiki_page = wikipedia.page("Apple", auto_suggest=False, redirect=False)
    dict = {"wiki_page": wiki_page.pageid, "final_similarity_index": 1}
    return jsonify(dict)


if __name__ == "__main__":
    # start flask app
    app.run(host="0.0.0.0", port=5000)
