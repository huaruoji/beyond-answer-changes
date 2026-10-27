from scripts.evaluate_manifest import PROVENANCE_FIELDS, parse_choice, style_question


def test_parse_choice_prefers_explicit_final_answer():
    assert parse_choice("The first sounds earlier. Answer: B") == "B"
    assert parse_choice("The answer is A, although B is plausible.") == "A"


def test_parse_choice_fallback_prose():
    assert parse_choice("the second event happens later") == "B"
    assert parse_choice("uncertain") is None


def test_parse_choice_ignores_echoed_chat_prompt():
    echoed = "system\nYou are helpful.\nuser\nAnswer A or B.\nassistant\nB"
    assert parse_choice(echoed) == "B"
    cot = (
        "user\nEnd with Answer: A or Answer: B.\nassistant\n"
        "A. The first event occurs earlier in the audio."
    )
    assert parse_choice(cot) == "A"


def test_parse_choice_does_not_treat_article_as_option():
    assert parse_choice("There is a quiet sound, but I am uncertain.") is None


def test_parse_choice_accepts_verbose_answer_prefix():
    assert parse_choice("Answer with exactly A or B: A") == "A"


def test_prompt_styles():
    q = "Answer A or B."
    assert style_question(q, "direct") == q
    assert "timestamps" in style_question(q, "timestamp")


def test_stratification_fields_are_preserved():
    """Dropping one of these silently destroys stratified analyses."""
    for field in ("source_pair_id", "surface_order", "option_events", "difficulty", "noise_snr_db"):
        assert field in PROVENANCE_FIELDS


def test_synth_provenance_fields_are_preserved():
    """Screening and attribution both read these; dropping one is silent."""
    for field in (
        "query_class",
        "absent_class",
        "edited_span_s",
        "split",
        "snr_db",
        "background_class",
    ):
        assert field in PROVENANCE_FIELDS


def test_default_conditions_are_not_hardcoded():
    """A fixed default silently dropped five of seven conditions on a real run.

    The evaluator must default to whatever the manifest actually contains, so a
    manifest built for a different condition set cannot be quietly narrowed.
    """
    import inspect

    from scripts import evaluate_manifest

    source = inspect.getsource(evaluate_manifest.main)
    assert '"--conditions"' in source
    assert "default=None" in source
    assert "args.conditions is None" in source
