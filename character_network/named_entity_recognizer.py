import spacy
from nltk import sent_tokenize
from ast import literal_eval
import pandas as pd
import os
import sys
import pathlib
folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_subtitles_dataset

class NamedEntityRecognizer:
    def __init__(self):
        self.nlp_model = self.load_model()
        pass


    def load_model(self):
        nlp = spacy.load("en_core_web_trf")
        return nlp
    

    def get_ner_output(self, script):
        script_sentences = sent_tokenize(script)

        ner_output = []

        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ == "PERSON":
                    entity_full_name = entity.text
                    entity_first_name = entity_full_name.split(" ")[0]
                    entity_first_name = entity_first_name.strip()
                    ners.add(entity_first_name)
            ner_output.append(ners)
        
        return ner_output
    

    def get_ners(self, dataset_path, save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ner'] = df['ner'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df

        # Load dataset
        df = load_subtitles_dataset(dataset_path)

        # Run inference
        df['ner'] = df['script'].apply(self.get_ner_output)

        if save_path is not None:
            df.to_csv(save_path, index=False)
        
        return df