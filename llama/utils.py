def determine_tokenizer_truncation(
    tokenizer,
    config,
    truncation_len=None,
    chunking_len=None
):

    """
    Determines whether tokenizer truncation should be enabled, and what max_length
    should be passed to the tokenizer.

    This function considers the user-specified `truncation_len` and `chunking_len`, and
    compares them with the model's maximum supported input length.

    Args:
        tokenizer (PreTrainedTokenizer): 
            Hugging Face tokenizer object, used to fetch `model_max_length`.
        config (PretrainedConfig): 
            Model config with `max_position_embeddings`.
        truncation_len (int, optional): 
            If provided, this is the length to truncate each input sequence to.
        chunking_len (int, optional): 
            If provided, this is the length to chunk long inputs into fixed-size parts.

    Returns:
        tuple:
            - should_truncate (bool): Whether truncation should be applied in tokenizer().
            - tokenizer_max_len (int or None): Value to use for `max_length`. Can be None if no truncation needed.
    """

    should_truncate = False
    tokenizer_max_len = None

    model_max_len = min(tokenizer.model_max_length, config.max_position_embeddings)

    print(f"✂️ User_specified truncation len: {truncation_len}, chunking len: ({chunking_len}), model max len: {model_max_len}")

    if not truncation_len and not chunking_len:
        should_truncate = True
        tokenizer_max_len = model_max_len
        print(f"No truncation or chunking specified. Tokenizer will truncate to model_max_len ({tokenizer_max_len})")

    elif truncation_len and not chunking_len:
        should_truncate = True
        tokenizer_max_len = min(model_max_len, truncation_len)
        print(f"Tokenizer will truncate to min(model_max_len, truncation_len) ({tokenizer_max_len}). No chunking used")

    elif truncation_len and chunking_len:
        should_truncate = True
        user_specified_max_len = min(truncation_len, chunking_len)

        if user_specified_max_len > model_max_len:
            tokenizer_max_len = model_max_len
            print(f"user_specified_max_len ({user_specified_max_len}) > model_max_len ({chunking_len}). Tokenizer will truncate to model_max_len ({tokenizer_max_len})")
        else:
            tokenizer_max_len = truncation_len
            print(f"Tokenizer will truncate to truncation_len ({tokenizer_max_len})")

    elif not truncation_len and chunking_len:
        if chunking_len > model_max_len:
            should_truncate = True
            tokenizer_max_len = model_max_len
            print(f"chunking_len ({chunking_len}) > model_max_len ({model_max_len}). Tokenizer will truncate to model_max_len ({tokenizer_max_len})")
        else:
            should_truncate = False
            tokenizer_max_len = None
            print(f"chunking_len ({chunking_len}) used. No truncation.")
    
    return should_truncate, tokenizer_max_len