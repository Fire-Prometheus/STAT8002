import pandas as pd
from python.preprocessing import GRAINS

headline = pd.read_pickle('../data/preprocessed_all_new_tags_headline.pickle')
content = pd.read_pickle('../data/preprocessed_all_new_tags_content.pickle')


def contain_keywords(tag: str, grain: str) -> bool:
    return grain.lower() in tag.lower()


def count_original(df: pd.DataFrame) -> None:
    for grain in GRAINS:
        print(grain)
        print(len(df[df['tags'].apply(
            lambda tag: contain_keywords(tag, grain) if type(tag) is str and len(tag) > 0 else False)]))


def count_new(df: pd.DataFrame) -> None:
    for grain in GRAINS:
        print(grain)
        print(len(df[df['new tags'].apply(lambda tag: grain.lower() in tag)]))


print('=============================')
print('original')
print(count_original(headline))
print('new')
print(count_new(headline))
print('=============================')
print('original')
print(count_original(content))
print('new')
print(count_new(content))
