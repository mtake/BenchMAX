import pandas as pd

def main():
    model = "qwen2.5-72b-chat"
    benchmark = "synthetic"
    ctx_lens = [131072]
    langs = ["en", "zh", "es", "fr", "de", "ru", "ja", "th", "sw", "bn", "te", "ar", "ko", "vi", "cs", "hu", "sr"]

    for lang in langs:
        for l in ctx_lens:
            print(lang, l)
            summary = pd.read_csv(f"../results/{model}/{benchmark}/{lang}/{l}/pred/summary.csv")
            scores = summary.iloc[1][1:].tolist()
            print("\t".join(scores) + "\n")
    

if __name__ == "__main__":
    main()
