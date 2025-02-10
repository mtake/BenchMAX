import sacrebleu

def get_spBLEU(hyps, refs):
    assert len(hyps) == len(refs[0]), "The number of translations should be equal to the number of references"
    result = sacrebleu.corpus_bleu(hyps, refs, tokenize="flores200", force=True)
    return result.score

def get_BLEU(hyps, refs, tgt_lang):
    assert len(hyps) == len(refs[0]), "The number of translations should be equal to the number of references"
    tokenizer = "13a"
    if tgt_lang == "zh":
        tokenizer = "zh"
    elif tgt_lang == "ja":
        tokenizer = "ja-mecab"
    elif tgt_lang == "ko":
        tokenizer = "ko-mecab"
    result = sacrebleu.corpus_bleu(hyps, refs, tokenize=tokenizer, force=True)
    return result.score

def get_chrf(hyps, refs):
    assert len(hyps) == len(refs[0]), "The number of translations should be equal to the number of references"
    result = sacrebleu.corpus_chrf(hyps, refs, remove_whitespace=False)
    return result.score

def get_ter(hyps, refs):
    assert len(hyps) == len(refs[0]), "The number of translations should be equal to the number of references"
    result = sacrebleu.corpus_ter(hyps, refs, normalized=True, asian_support=True)
    return result.score

def _get_xComet():
    comet_model = None

    def inner(srcs, hyps, refs):
        nonlocal comet_model
        assert len(srcs) == len(hyps) == len(refs[0]), "The number of translations should be equal to the number of references"
        from comet import load_from_checkpoint, download_model
        if comet_model is None:
            model_path = download_model("Unbabel/XCOMET-XXL")
            comet_model = load_from_checkpoint(model_path, local_files_only=True)
        data = [{"src": src, "ref": ref, "mt": hyp} for src, hyp, ref in zip(srcs, hyps, refs[0])]
        prediction = comet_model.predict(data, batch_size=16, gpus=1)
        return prediction.system_score

    return inner

get_xComet = _get_xComet()