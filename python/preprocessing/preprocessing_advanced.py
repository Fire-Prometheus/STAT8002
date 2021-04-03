import operator
from typing import List, Set, Tuple

from python.preprocessing import AdvancedNewsDataPreprocessor, PICKLE, GRAINS
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# advanced_data_preprocessor = AdvancedNewsDataPreprocessor('../data/preprocessed_news_all_new.pickle')
# model = advanced_data_preprocessor.model
# df = advanced_data_preprocessor.data_frame

df = pd.read_pickle('../data/preprocessed_news_all_new_tags.pickle')
doc_vec = Doc2Vec.load('../data/doc2vec_content.model')
no_tags_df = df[df.swifter.apply(lambda row: len(row['new tags']) == 0, axis=1)]

MY_GRAINS = list(map(lambda g: g.lower(), GRAINS))
MY_GRAINS.append('soy')


def predict_topics(content: List[str]) -> List[str]:
    existing_grains: Set[str] = set(filter(lambda g: g in content, MY_GRAINS))
    if len(existing_grains) == 0:
        return []
    infer_vector = doc_vec.infer_vector(content)
    most_similar: List[Tuple[str, float]] = doc_vec.docvecs.most_similar([infer_vector])
    most_similar = list(filter(lambda t: operator.itemgetter(1)(t) > 0, most_similar))
    most_similar_topics: Set[str] = set(map(lambda t: operator.itemgetter(0)(t), most_similar))
    return list(most_similar_topics.intersection(existing_grains))


no_tags_df = no_tags_df.drop(columns=['new tags'])
no_tags_df['new tags'] = no_tags_df['new content'].swifter.apply(lambda c: predict_topics(c))

df_new = df[df.swifter.apply(lambda row: len(row['new tags']) > 0, axis=1)]
result = pd.concat([df_new, no_tags_df]).sort_index()
result.to_pickle('../data/preprocessed_all_new_tags_content.pickle')
