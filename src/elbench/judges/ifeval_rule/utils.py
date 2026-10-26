import dataclasses
from typing import Dict, Optional, Union

from . import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(
    inp,
    response,
):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and 'prompt' in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    response,
):
    """Tests response for an upper bound for following instructions."""
    r = response.split('\n')
    response_remove_first = '\n'.join(r[1:]).strip()
    response_remove_last = '\n'.join(r[:-1]).strip()
    response_remove_both = '\n'.join(r[1:-1]).strip()
    revised_response = response.replace('*', '')
    revised_response_remove_first = response_remove_first.replace('*', '')
    revised_response_remove_last = response_remove_last.replace('*', '')
    revised_response_remove_both = response_remove_both.replace('*', '')
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and 'prompt' in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    inp = InputExample(
        key=doc['key'],
        instruction_id_list=doc['instruction_id_list'],
        prompt=doc['prompt'],
        kwargs=doc['kwargs'],
    )
    response = results[0]

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        'prompt_level_strict': float(out_strict.follow_all_instructions),
        'inst_level_strict': agg_inst_level_acc(out_strict.follow_instruction_list),
        'prompt_level_loose': float(out_loose.follow_all_instructions),
        'inst_level_loose': agg_inst_level_acc(out_loose.follow_instruction_list),
    }


def agg_inst_level_acc(items):
    inst_level_acc = sum(items) / len(items) if items else 0
    return inst_level_acc


def main():
    doc = {'instruction_id_list': ['punctuation:no_comma', 'detectable_format:number_highlighted_sections', 'length_constraints:number_words'], 'key': 1000, 'kwargs': [{'capital_frequency': None, 'capital_relation': None, 'end_phrase': None, 'first_word': None, 'forbidden_words': None, 'frequency': None, 'keyword': None, 'keywords': None, 'language': None, 'let_frequency': None, 'let_relation': None, 'letter': None, 'nth_paragraph': None, 'num_bullets': None, 'num_highlights': None, 'num_paragraphs': None, 'num_placeholders': None, 'num_sections': None, 'num_sentences': None, 'num_words': None, 'postscript_marker': None, 'prompt_to_repeat': None, 'relation': None, 'section_spliter': None}, {'capital_frequency': None, 'capital_relation': None, 'end_phrase': None, 'first_word': None, 'forbidden_words': None, 'frequency': None, 'keyword': None, 'keywords': None, 'language': None, 'let_frequency': None, 'let_relation': None, 'letter': None, 'nth_paragraph': None, 'num_bullets': None, 'num_highlights': 3, 'num_paragraphs': None, 'num_placeholders': None, 'num_sections': None, 'num_sentences': None, 'num_words': None, 'postscript_marker': None, 'prompt_to_repeat': None, 'relation': None, 'section_spliter': None}, {'capital_frequency': None, 'capital_relation': None, 'end_phrase': None, 'first_word': None, 'forbidden_words': None, 'frequency': None, 'keyword': None, 'keywords': None, 'language': None, 'let_frequency': None, 'let_relation': None, 'letter': None, 'nth_paragraph': None, 'num_bullets': None, 'num_highlights': None, 'num_paragraphs': None, 'num_placeholders': None, 'num_sections': None, 'num_sentences': None, 'num_words': 300, 'postscript_marker': None, 'prompt_to_repeat': None, 'relation': 'at least', 'section_spliter': None}], 'prompt': 'Write a 300+ word summary of the wikipedia page "https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.'}

    results = ['''Raymond III Count of Tripoli was a significant figure in the Latin East. He was born around 1140. His father Raymond II was murdered in 1152. Raymond III was then a young boy. King Baldwin III of Jerusalem became his first regent. Later King Amalric I of Jerusalem served as his guardian. Raymond married Eschiva of Bures in 1174. She was Princess of Galilee. This marriage brought him great wealth and extensive lands. He became the most powerful baron in the Crusader states. His influence extended throughout the kingdom.

*Regency and Political Maneuvering*
King Amalric I died in 1174. His son Baldwin IV became king. Baldwin IV was a minor and suffered from leprosy. Raymond III was appointed regent for the young king. He held this crucial position until 1177. During his regency Raymond negotiated a truce with Saladin. This provided a period of peace for the kingdom. He also released many prisoners including Bohemond III of Antioch. Raymond tried to arrange a suitable marriage for Sibylla Baldwin IV's sister. He aimed to secure a strong successor for the throne. He sought a European nobleman for Sibylla. He wanted a leader capable of defending the kingdom. His efforts were often thwarted by internal politics. Baldwin IV reached his majority. Raymond's regency ended. However he remained a key advisor. His political acumen was highly valued. He always sought the kingdom's stability.

*Rivalry with Guy of Lusignan*
Baldwin IV's leprosy worsened over time. He needed a co-ruler or an heir. Guy of Lusignan arrived in the kingdom. He was a French nobleman. Guy secretly married Sibylla in 1180. This marriage deeply angered Raymond. Raymond considered Guy an unsuitable and dangerous choice. He feared Guy's impulsive nature. Raymond then supported Baldwin V Sibylla's son as heir. He tried to prevent Guy from gaining more power. The kingdom became divided into two factions. Raymond led the baronial party. Guy led a more aggressive court party. This internal conflict weakened the Crusader states. Raymond even formed a temporary alliance with Saladin. He did this to counter Guy's provocations. This move was controversial. It showed his desperation to control Guy's recklessness. He believed Guy would destroy the kingdom.

*The Disaster of Hattin and Final Days*
Baldwin IV died in 1185. Baldwin V died shortly after in 1186. Guy of Lusignan then became king. Sibylla was crowned queen. Guy immediately broke truces with Saladin. Saladin prepared for a major invasion. Raymond advised caution and defensive strategy. He urged Guy not to engage Saladin in open battle. Guy ignored Raymond's wise counsel. The Crusader army marched to relieve Tiberias. Raymond's wife Eschiva was besieged there. The army was exhausted and thirsty. They were trapped near the Horns of Hattin. Raymond's division formed the vanguard. During the battle Raymond's men broke through Saladin's lines. He escaped the catastrophic defeat. Many Crusaders were killed or captured. This battle destroyed the main Crusader army. Raymond retreated to Tripoli. He was deeply distraught by the loss. He died of pleurisy in August 1187. He named his nephew Raymond IV of Antioch as his successor. Raymond III was a complex leader. He was a skilled diplomat and military strategist. He consistently tried to preserve the Crusader states. His efforts were ultimately overwhelmed by internal divisions and Saladin's strength. His death marked a profound loss for the Crusader cause.''']
    score = process_results(doc, results=results)

    print('*' * 50 + f'''\n{score}\n^^^()^^^\n''' + '''\nat:\ngit_repos/evalscope_driver/evalscope_1_4_2/evalscope/benchmarks/ifeval/utils.py:150\n''' + '*' * 50)

if __name__ == '__main__':
    main()
