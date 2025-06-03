# python3 metrics.py --pred_file pred_base.txt --ref_file /nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024/wmt2024.test.raw.en --output bleu.base
# python3 metrics.py --pred_file predictions-128000.txt --ref_file /nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024/wmt2024.test.raw.en --output bleu.ft.128000
# python3 metrics.py --pred_file predictions-138500.txt --ref_file /nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024/wmt2024.test.raw.en --output bleu.ft.138500
# python3 metrics.py --pred_file predictions-141000.txt --ref_file /nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024/wmt2024.test.raw.en --output bleu.ft.141000
# python3 metrics.py --pred_file predictions-133000.txt --ref_file /nfs/RESEARCH/avila/Challenges/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024/wmt2024.test.raw.en --output bleu.ft.133000
# python3 metrics.py --pred_file /nfs/RESEARCH/avila/Challenges/WMT2024_NON-REPETITIVE/predictions-133000_ft.txt --ref_file /nfs/RESEARCH/avila/Challenges/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024/wmt2024.test.raw.en --output bleu.ft.only.133000
# python3 metrics.py --pred_file /nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/llama.en --ref_file /nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024/wmt2024.test.raw.en --output bleu.llama
# python3 metrics.py --pred_file /nfs/RESEARCH/avila/Challenges/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024/nllb-pred.txt --ref_file /nfs/RESEARCH/avila/Challenges/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024/wmt2024.test.raw.en --output bleu.nllb
python3 metrics.py --pred_file /nfs/RESEARCH/avila/Challenges/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024/pred-gpt.txt --ref_file /nfs/RESEARCH/avila/Challenges/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024/wmt2024.test.raw.en --output bleu.gpt
