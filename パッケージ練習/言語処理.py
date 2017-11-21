# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 18:38:15 2017

"""
from janome.tokenizer import Tokenizer
t = Tokenizer()
word = (u'私たちが希求するものは、党の利益ではなく、議員の利益でもなく、国民のため、つまり国民が納める税の恩恵を全ての国民に届ける仕組みを強化することにある。国政を透明化し、常に情報を公開し、国民と共にすすめる政治を実現する。既得権益、しがらみ、不透明な利権を排除し、 国民ファーストな政治を実現する。国民ひとりひとりに、日本に、未来に、希望を生むために。')
tokens = t.tokenize(word,wakati=False)
for token in tokens:
    print(token)
    
for token in tokens:
    partOfSpeech = token.part_of_speech.split(',')[0]
    if partOfSpeech == u'名詞':
        print(token.surface)
