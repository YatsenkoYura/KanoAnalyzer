from module.text_processor import TextProcessor

if __name__ == "__main__":
    text_processor = TextProcessor("Мама у моря мыла тару, а я отдыхал")
    #text_processor.tokenize()
    #text_processor.normalize()
    #text_processor.erase_stop_word()
    #text_processor.lematize()
    print(text_processor.apply_text_processing())
