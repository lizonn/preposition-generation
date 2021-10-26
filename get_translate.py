import requests
from english.BERT.config import deepl_token


def translate(text):
    url = 'https://api.deepl.com/v2/translate'
    params = {'auth_key': deepl_token ,
              'text': text,
              "preserve_formatting":1,
              'formality':'less',
              'split_sentences':0,
              'target_lang': 'RU'}

    response = requests.get(url, params=params)

    if response.ok:
        print(response.json()['translations'][0]['text'])

    else:
        print('Sorry, we can`t found')


if __name__ == '__main__':
    text = "make French toast."
    # text = 'hold  angle tooth with volcano'
    translate(text)