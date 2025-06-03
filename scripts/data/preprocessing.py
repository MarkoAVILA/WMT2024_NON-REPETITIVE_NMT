import pyonmttok as tok_opennmt
import fire
import logging
import json

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


class Preprocessing():
    def __init__(self, n_symbols=32000, vocab_path=" ", model_path='', config=" ", from_systran=True) -> None:
        self.n_symbols = n_symbols
        self.model_path = model_path
        self.config = config
        self.vocab_path = vocab_path
        self.from_systran =  from_systran
    
    # def load_config_tokenizer(self):
    #     b = json.load(open(self.config))
    #     for key, b_value in b.items():
    #         if key=="preprocess" and not isinstance(b_value, dict):
    #             for j in b_value:
    #                 if j['op']=="tokenization" and j['name']=="tokenization" :
    #                     res = j
    #     return res 


    def building_tokenizer_model(self, *files):
        self.files = files
        tokenizer = tok_opennmt.Tokenizer("aggressive", joiner_annotate=True, segment_numbers=True)
        learner = tok_opennmt.BPELearner(tokenizer=tokenizer, symbols=self.n_symbols)
        logging.info("Building BPE Model ...")
        for file in self.files:
            learner.ingest_file(file)
        self.tokenizer = learner.learn(self.model_path)
        logging.info("BPE Model is ready for using it!")


    def building_vocabulaire(self, name):
        logging.info("Building Vocabulary...")
        lines_ = []
        for file in self.files:
            with open(file,encoding='utf-8') as f:
                lines = [i.strip() for i in f]
            lines_.extend(lines)
        
        self.vocab = tok_opennmt.build_vocab_from_lines(lines_, 
                                                 tokenizer=self.tokenizer, 
                                                 maximum_size=self.n_symbols, 
                                                 special_tokens=["<blank>","<unk>", "<s>", "</s>"]
                                                 )
        with open(name, "w+", encoding="utf-8") as vocab_file:
            for token in self.vocab.ids_to_tokens:
                vocab_file.write("%s\n" % token)
        logging.info("Vocabulary is ready to use it!")

    def tokenization_from_file(self, input_file, output_file):
        self.tokenizer.tokenize_file(input_file, output_file)
        logging.info('Tokenization done!')

    def tokenization(self, input):
        output=[" ".join(self.tokenizer.tokenize(i.strip())[0]) for i in input]
        logging.info('Tokenization done!')
        return output
    
    def tokenization_bpe_path(self, ty, input_file, output_file):

        if self.from_systran:
            # config = self.load_config_tokenizer()
            # config[ty]["bpe_model_path"] = self.model_path
            # conf = config[ty]
            ## Config of tokenizer for Systran Model
            tokenizer = tok_opennmt.Tokenizer(mode="aggressive", #conf.get("mode"), 
                                              joiner_annotate=True, #conf.get("joiner_annotate"), 
                                              preserve_placeholders=True, #conf.get("preserve_placeholders"), 
                                              preserve_segmented_tokens=True, #conf.get("preserve_segmented_tokens"),  
                                              segment_case=True, #conf.get("segment_case"), 
                                              segment_numbers=True,
                                              case_markup= True,
                                              segment_alphabet_change=True,
                                              bpe_model_path=self.model_path, #conf.get("bpe_model_path"),
                                              vocabulary_path = self.vocab_path
                                              )
        else:
            tokenizer = tok_opennmt.Tokenizer("aggressive", joiner_annotate=True, segment_numbers=True, 
                                              bpe_model_path=self.model_path
                                              )

        tokenizer.tokenize_file(input_file, output_file)
        logging.info('Tokenization done!')

    def detokenization_from_file(self, file_input, file_output):
        self.tokenizer.detokenize_file(file_input, file_output)
        logging.info("Detokenization done!")

    def detokenization_from_file_bpe_path(self, ty, file_input, file_output):
        if self.from_systran:
            # config = self.load_config_tokenizer()
            # config[ty]["bpe_model_path"] = self.model_path
            # conf = config[ty]
            ## Config of tokenizer for Systran Model
            tokenizer = tok_opennmt.Tokenizer(mode="aggressive", #conf.get("mode"), 
                                              joiner_annotate=True, #conf.get("joiner_annotate"), 
                                              preserve_placeholders=True, #conf.get("preserve_placeholders"), 
                                              preserve_segmented_tokens=True, #conf.get("preserve_segmented_tokens"),  
                                              segment_case=True, #conf.get("segment_case"), 
                                              segment_numbers=True,
                                              case_markup= True,
                                              segment_alphabet_change=True,
                                              bpe_model_path=self.model_path, #conf.get("bpe_model_path"),
                                              vocabulary_path = self.vocab_path
                                              )
        else:
            tokenizer = tok_opennmt.Tokenizer("aggressive", joiner_annotate=True, segment_numbers=True, 
                                              bpe_model_path=self.model_path
                                              )
        
        tokenizer.detokenize_file(file_input, file_output)
        logging.info("Detokenization done!")

        
    def detokenization(self, input):
        tokenizer = tok_opennmt.Tokenizer(mode="aggressive", #conf.get("mode"), 
                                              joiner_annotate=True, #conf.get("joiner_annotate"), 
                                              preserve_placeholders=True, #conf.get("preserve_placeholders"), 
                                              preserve_segmented_tokens=True, #conf.get("preserve_segmented_tokens"),  
                                              segment_case=True, #conf.get("segment_case"), 
                                              segment_numbers=True,
                                              case_markup= True,
                                              segment_alphabet_change=True,
                                              bpe_model_path=self.model_path, #conf.get("bpe_model_path"),
                                              vocabulary_path = self.vocab_path
                                              )
        output = [tokenizer.detokenize(i.rstrip().split(' ')) for i in input]
        logging.info("Detokenization done!")
        return output

    def main(self, *files, name_vocab, src_file, tgt_file):
        self.building_tokenizer_model(*files)
        self.building_vocabulaire(*files, name_vocab)
        self.tokenization_from_file(src_file, src_file+'.tok')
        self.tokenization_from_file(tgt_file, tgt_file+'.tok')

if __name__=='__main__':
    fire.Fire(Preprocessing)



