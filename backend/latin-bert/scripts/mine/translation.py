from latinTranslator import LatinTranslator

translator = LatinTranslator()  
latin_sentence = "Despectus tibi sum nec"
english_translation = translator.translate(latin_sentence) 
print("This is the translated word:", english_translation)
