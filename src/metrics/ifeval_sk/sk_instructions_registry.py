# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of all Slovak instructions."""

import src.metrics.ifeval_sk.sk_instructions as sk_instructions


_KEYWORD = "keywords:"

_LANGUAGE = "language:"

_LENGTH = "length_constraints:"

_CONTENT = "detectable_content:"

_FORMAT = "detectable_format:"

_MULTITURN = "multi-turn:"

_COMBINATION = "combination:"

_STARTEND = "startend:"

_CHANGE_CASES = "change_case:"

_PUNCTUATION = "punctuation:"

INSTRUCTION_DICT = {
    _KEYWORD + "existence": sk_instructions.KeywordChecker,
    _KEYWORD + "frequency": sk_instructions.KeywordFrequencyChecker,
    _KEYWORD + "forbidden_words": sk_instructions.ForbiddenWords,
    _KEYWORD + "letter_frequency": sk_instructions.LetterFrequencyChecker,
    _LANGUAGE + "response_language": sk_instructions.ResponseLanguageChecker,
    _LENGTH + "number_sentences": sk_instructions.NumberOfSentences,
    _LENGTH + "number_paragraphs": sk_instructions.ParagraphChecker,
    _LENGTH + "number_words": sk_instructions.NumberOfWords,
    _LENGTH + "nth_paragraph_first_word": sk_instructions.ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": sk_instructions.PlaceholderChecker,
    _CONTENT + "postscript": sk_instructions.PostscriptChecker,
    _FORMAT + "number_bullet_lists": sk_instructions.BulletListChecker,
    _FORMAT + "constrained_response": sk_instructions.ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": sk_instructions.HighlightSectionChecker,
    _FORMAT + "multiple_sections": sk_instructions.SectionChecker,
    _FORMAT + "json_format": sk_instructions.JsonFormat,
    _FORMAT + "title": sk_instructions.TitleChecker,
    _COMBINATION + "two_responses": sk_instructions.TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": sk_instructions.RepeatPromptThenAnswer,
    _STARTEND + "end_checker": sk_instructions.EndChecker,
    _CHANGE_CASES + "capital_word_frequency": sk_instructions.CapitalWordFrequencyChecker,
    _CHANGE_CASES + "slovak_capital": sk_instructions.CapitalLettersSlovakChecker,
    _CHANGE_CASES + "slovak_lowercase": sk_instructions.LowercaseLettersSlovakChecker,
    _PUNCTUATION + "no_comma": sk_instructions.CommaChecker,
    _STARTEND + "quotation": sk_instructions.QuotationChecker,
}

INSTRUCTION_CONFLICTS = {
    _KEYWORD + "existence": {_KEYWORD + "existence"},
    _KEYWORD + "frequency": {_KEYWORD + "frequency"},
    _KEYWORD + "forbidden_words": {_KEYWORD + "forbidden_words"},
    _KEYWORD + "letter_frequency": {_KEYWORD + "letter_frequency"},
    _LANGUAGE + "response_language": {
        _LANGUAGE + "response_language",
        _FORMAT + "multiple_sections",
        _KEYWORD + "existence",
        _KEYWORD + "frequency",
        _KEYWORD + "forbidden_words",
        _STARTEND + "end_checker",
        _CHANGE_CASES + "slovak_capital",
        _CHANGE_CASES + "slovak_lowercase",
    },
    _LENGTH + "number_sentences": {_LENGTH + "number_sentences"},
    _LENGTH + "number_paragraphs": {
        _LENGTH + "number_paragraphs",
        _LENGTH + "nth_paragraph_first_word",
        _LENGTH + "number_sentences",
    },
    _LENGTH + "number_words": {_LENGTH + "number_words"},
    _LENGTH + "nth_paragraph_first_word": {
        _LENGTH + "nth_paragraph_first_word",
        _LENGTH + "number_paragraphs",
    },
    _CONTENT + "number_placeholders": {_CONTENT + "number_placeholders"},
    _CONTENT + "postscript": {_CONTENT + "postscript"},
    _FORMAT + "number_bullet_lists": {_FORMAT + "number_bullet_lists"},
    _FORMAT + "constrained_response": set(INSTRUCTION_DICT.keys()),
    _FORMAT + "number_highlighted_sections": {_FORMAT + "number_highlighted_sections"},
    _FORMAT + "multiple_sections": {
        _FORMAT + "multiple_sections",
        _LANGUAGE + "response_language",
        _FORMAT + "number_highlighted_sections",
    },
    _FORMAT + "json_format": set(INSTRUCTION_DICT.keys()).difference(
        {_KEYWORD + "forbidden_words", _KEYWORD + "existence"}
    ),
    _FORMAT + "title": {_FORMAT + "title"},
    _COMBINATION + "two_responses": set(INSTRUCTION_DICT.keys()).difference(
        {
            _KEYWORD + "forbidden_words",
            _KEYWORD + "existence",
            _LANGUAGE + "response_language",
            _FORMAT + "title",
            _PUNCTUATION + "no_comma",
        }
    ),
    _COMBINATION + "repeat_prompt": set(INSTRUCTION_DICT.keys()).difference(
        {_KEYWORD + "existence", _FORMAT + "title", _PUNCTUATION + "no_comma"}
    ),
    _STARTEND + "end_checker": {_STARTEND + "end_checker"},
    _CHANGE_CASES + "capital_word_frequency": {
        _CHANGE_CASES + "capital_word_frequency",
        _CHANGE_CASES + "slovak_lowercase",
        _CHANGE_CASES + "slovak_capital",
    },
    _CHANGE_CASES + "slovak_capital": {_CHANGE_CASES + "slovak_capital"},
    _CHANGE_CASES + "slovak_lowercase": {
        _CHANGE_CASES + "slovak_lowercase",
        _CHANGE_CASES + "slovak_capital",
    },
    _PUNCTUATION + "no_comma": {_PUNCTUATION + "no_comma"},
    _STARTEND + "quotation": {_STARTEND + "quotation", _FORMAT + "title"},
}


def conflict_make(conflicts):
    """Makes sure if A conflicts with B, B will conflict with A.

    Args:
      conflicts: Dictionary of potential conflicts where key is instruction id
        and value is set of instruction ids that it conflicts with.

    Returns:
      Revised version of the dictionary. All instructions conflict with
      themselves. If A conflicts with B, B will conflict with A.
    """
    for key in conflicts:
        for k in conflicts[key]:
            conflicts[k].add(key)
        conflicts[key].add(key)
    return conflicts
