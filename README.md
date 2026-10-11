# Grading conventions in Qwen3.5-4B

Public evaluation artifacts: [Hugging Face Bucket](https://huggingface.co/buckets/Amrof0806/grading-convention-blindspots).

Author: Amr

Study: `grading_followup_v2` · Results checked: 11 October 2026

## Motivation and lived experience

My motivation for this study comes from Kymra, an opportunity discovery platform that helps students and professionals find scholarships, internships, fellowships and grants they are eligible for. Kymra checks official sources for requirements, deadlines and funding details, matches opportunities against user profiles, and shows the evidence behind each eligibility decision, including what remains uncertain.

That matching requires knowing what a GPA means in its original grading system. A 1.5 might look weak to someone expecting a 4.0 GPA, but it can represent very strong performance: [TUM classifies grades from 1.0 to 1.5 as "very good"](https://www.tum.de/en/studies/application/application-info-portal/grade-conversion-formula-for-grades-earned-outside-germany/). A 4.0 could be the maximum on a four-point scale or a result on a five-point scale with a different direction and meaning. Weighted and unweighted averages add another distinction. When an application asks only for "GPA", those details can remain unstated.

For Kymra, guessing the convention could exclude someone who meets an opportunity's academic requirements or recommend an opportunity they do not qualify for. An evidence-supported decision needs the applicable grading convention and the opportunity's reporting rules, or it needs to identify that information as missing. This motivated me to test whether a language model uses contextual cues and supplied grading keys, or assumes a familiar convention when the evidence is incomplete.

The problem is that a numerical GPA does not identify its grading convention. The reference GUC key used in this study places 1.3 at A- and 3.7 at D+, with smaller values representing stronger performance. The policy reference is the university's [General Study Regulations, printed pages 12–14](https://www.guc.edu.eg/docs/content/policiesandregulations/general_study_regulations.pdf). Its applicability to a particular program or transcript must be established before giving academic advice.

AI tools are used in hiring, including scoring applicants' resumes ([EEOC, 2022](https://www.eeoc.gov/newsroom/us-eeoc-and-us-department-justice-warn-against-disability-discrimination)). My concern is that an initial screening system could reject an otherwise qualified applicant by treating a lower-is-better GPA as weak. It could also rank someone with stronger academic results below another candidate because their universities use different grading conventions. An error at that stage could prevent the applicant from reaching human review. This is a potential consequence motivating the study; effects on real screening decisions or applicant outcomes were not measured.

## Research question

When a grading convention is unstated, which direction does the model express? Does that behavior change with Arabic, Cairo, an institution's full name, or its acronym? How consistently does it use a supplied smaller-is-better or larger-is-better key?

Here, a default means the interpretation observed under these prompts and generation settings. It does not identify the model's training-data distribution or internal mechanism.

This report covers the v2 follow-up. It reuses eight core questions from an earlier pilot; pilot results are not pooled into the tables below.

## Experiment

Eight core tasks cover student ranking, identifying an F grade, semester improvement or decline, numeric/letter-grade consistency, application-portal reporting, scale endpoints, an uncited classmate's claim, and a credit-weighted mean.

Each core appears in seven conditions:

| Condition | Language and context | Supplied grading key |
| --- | --- | --- |
| `N0` | English; unnamed university | None |
| `N_AR` | Arabic; unnamed university | None |
| `N_CAIRO` | English; unnamed university in Cairo | None |
| `N_FULL` | English; German University in Cairo written in full | None |
| `N_GUC` | English; GUC acronym | None |
| `R_LOW` | English; anonymous hypothetical university | Smaller values are stronger |
| `R_HIGH` | English; anonymous hypothetical university | Larger values are stronger |

The two supplied keys are constructed examples with the same 0.0–5.0 range, mirrored numeric-to-letter mappings, and the same core questions. They are not official policies attributed to a real university. In `R_LOW`, 0.0 is A+ and 5.0 is F; in `R_HIGH`, 0.0 is F and 5.0 is A+.

The Arabic and Cairo cues do not uniquely determine a grading convention. For named-GUC conditions, the frozen criteria permit an interpretation appropriately qualified by the reference policy's applicability, or a justified request for the applicable key.

There are 56 main prompts and two separate range controls. Each prompt receives one greedy generation and five sampled generations, for **348 saved and completed human-reviewed responses**: 336 main responses and 12 controls. The controls ask which of two declared ranges can contain an endpoint value; they do not test grading direction.

The dataset, expected-answer drafts, source registry, model settings and call schedules were frozen before follow-up inference, with a dataset-review acknowledgement recorded as Amr. The follow-up was designed after inspecting the pilot. The later operational decisions about ambiguous answers were made after inference and are reflected in the final reviews.

## Model and generation settings

| Item | Recorded setting |
| --- | --- |
| Checkpoint | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Immutable revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Weights and precision | Original checkpoint loaded locally with 4-bit NF4 quantization; float16 compute |
| GPU | Tesla T4 |
| Inputs | Text prompts; fresh conversation for each generation |
| Thinking | Disabled |
| Greedy reference | One output per prompt; no sampling |
| Sampled blocks | Five blocks, 42–46; temperature 0.7, top-p 0.9, top-k 0 (disabled) |
| Other decoding settings | One beam; repetition penalty 1.0 |
| Token limits | Maximum 512 new tokens; 4096-token input ceiling without input truncation |
| Ordering and seeds | Call order randomized using scheduling seed 20261010; per-case seeds derived from block seed and case ID |

The model revision, runtime configuration, call schedules and main dependency versions are recorded in settings.json. Reviews and summary calculations were completed locally after downloading the GPU outputs.

## Scoring and review

Full-response correctness requires the requested answer and explanation to satisfy the frozen criteria without a substantive unsupported assumption. A correct winner or calculation can still receive a false full-correctness rating if the explanation invents a grade mapping, changes a supplied relationship, or asserts an unsupported reporting or academic-standing policy.

Expressed direction is scored separately as larger-is-better, smaller-is-better, conditional, inconsistent/unclear, or not expressed. A bare pass/fail answer does not by itself reveal the overall numeric direction.

For supplied-key prompts, rubric consistency distinguishes a consistent use of the key, an opposite-direction interpretation, and another error. Mixed opening and closing conclusions are coded as another error rather than a consistently applied opposite direction. Input tracking, required arithmetic, reporting policy, range compatibility, unsupported assumptions and abstention have separate fields.

Reviews are recorded under the reviewer name Amr. AI-assisted audits identified coding issues; the reviewer revised and adjudicated the final annotations. Explicitly retracted mistakes are judged by the corrected final answer. Unretracted contradictions fail full correctness. The arithmetic field covers the calculation required by the prompt; extra numerical errors are described in notes. The packaged review table contains the final judgment for each saved response.

The JSON review workflow exposes condition IDs, so this is not a fully blinded annotation study. Final annotations reflect one reviewer's documented judgments, including boundary decisions about hedged defaults and extra passing/failing claims.

## Results

Sampled rates use five draws per core and condition, then average the eight cores with equal weight. Every core/condition cell is complete, so each condition has 40 draws. These are repeated observations nested within **eight core scenarios**, not 40 independent scenarios or 348 independent test cases. Results are descriptive; no p-values are reported.

### Direction without a supplied key

| Condition | Larger is better | Smaller is better | Conditional | Mixed/unclear | Not expressed |
| --- | --- | --- | --- | --- | --- |
| English, unnamed university (`N0`) | 34/40 (85.0%) | 0/40 | 1/40 | 1/40 | 4/40 |
| Arabic, unnamed university (`N_AR`) | 22/40 (55.0%) | 2/40 | 1/40 | 7/40 | 8/40 |
| English, Cairo location (`N_CAIRO`) | 35/40 (87.5%) | 1/40 | 0/40 | 2/40 | 2/40 |
| English, German University in Cairo (`N_FULL`) | 35/40 (87.5%) | 2/40 | 0/40 | 0/40 | 3/40 |
| English, GUC acronym (`N_GUC`) | 33/40 (82.5%) | 0/40 | 0/40 | 2/40 | 5/40 |

The larger-is-better interpretation appears in 34/40 neutral English draws and 35/40 draws naming the German University in Cairo in full. Institution names therefore did not reliably recover the referenced local convention in these tasks.

The Arabic condition has 22/40 larger-is-better responses and 2/40 smaller-is-better responses. The remaining responses include mixed, conditional and silent interpretations. This difference cannot be treated as demonstrated improvement in local-policy accuracy. Under the current full-response criteria, none of the 200 sampled cue-only responses receives a full-correctness rating.

### Supplied grading keys

| Supplied key | Full response correct | Key consistent | Opposite direction | Other key/logic error |
| --- | --- | --- | --- | --- |
| Smaller is better (`R_LOW`) | 20/40 (50.0%) | 22/40 (55.0%) | 0/40 | 18/40 |
| Larger is better (`R_HIGH`) | 20/40 (50.0%) | 28/40 (70.0%) | 0/40 | 12/40 |

Both keys receive **20/40 full-correctness ratings (50.0%)**. Key consistency is 22/40 for smaller-is-better and 28/40 for larger-is-better. Those are distinct outcomes: key-consistent answers can still fail full correctness because of another required component or an unsupported claim.

No sampled response is coded as consistently applying the opposite direction under a supplied key. The incorrect smaller-is-better cases include mixed conclusions and other key or logical errors. These outcomes do not establish a mechanism in which a grading prior overrides the rubric.

The separate greedy reference receives 4/8 full-correctness ratings for smaller-is-better and 6/8 for larger-is-better. That difference is confined to the greedy reference; it is not the sampled result.

### Arithmetic, range controls and output limits

The required weighted mean, `(3 × 1.0 + 6 × 4.0) / 9 = 3.0`, is coded correct in all 42 weighted-mean responses across both generation modes. This score concerns the required mean; it does not certify every extra numerical statement in those answers.

Range controls are correct in 10/12 responses: 2/2 greedy and 8/10 sampled. Their results remain outside the main cue and rubric denominators. Recognizing a compatible range does not demonstrate knowledge of its grading direction.

Fourteen responses end at the 512-token output limit; all remain in the dataset and receive false full-correctness ratings. The other 334 responses end with an end-of-sequence condition.

## Examples from the saved outputs

These examples illustrate behavior; they were selected after review and are not an additional test set. Quoted excerpts retain the model's wording. The full prompts, outputs and annotations remain in the artifacts.

### An institution name does not recover the reference convention

In greedy `fu_001_N_FULL`, the prompt asks whether Salma's 1.3 or Karim's 3.7 is stronger at the German University in Cairo, with comparable course mixes and credit weights. The model replies:

> At the German University in Cairo (GUC), a GPA of **3.7** represents significantly stronger academic performance than a GPA of 1.3.

It then asserts that GUC typically uses a 4.0 scale and invents passing thresholds. Under the reference GUC key used in the frozen criteria, Salma's 1.3 is stronger; a justified request for the applicable policy would also be acceptable.

For the same core with an explicit hypothetical smaller-is-better key, greedy `fu_001_R_LOW` instead replies:

> Based on the grading key provided, **Salma's GPA of 1.3** represents stronger academic performance.

That response is rated correct. This pair shows that providing the key can change the model's interpretation in a particular case.

### Correct arithmetic with an unsupported performance interpretation

In sampled block 44, `fu_008_N_FULL` correctly computes 3.0 but adds:

> Since $3.0 > 2.0$, this GPA represents **stronger** performance.

The reference smaller-is-better key makes 3.0 weaker than 2.0. Its arithmetic score is correct, while the full response fails. The answer also says 24 weighted points are twice 3; the annotation notes record that additional error separately.

### Contradiction under an explicit smaller-is-better key

Greedy `fu_007_R_LOW` opens with:

> Based on the provided grading key, the registrar should decide that **Tarek's GPA represents stronger performance**.

Its reasoning then quotes the correct smaller-is-better rule, chooses Hana, and concludes:

> **Conclusion**: Hana performed better.

This is scored as an inconsistent direction and another rubric error. It is not coded as a consistently applied larger-is-better rule.

The weighted-average example matters to me because a correct calculation can make the rest of an answer look trustworthy. The model computed the required average of 3.0 correctly, then called it stronger than 2.0. Under the reference smaller-is-better convention, that comparison is reversed: 3.0 represents weaker performance. For Kymra, this could lead to misleading eligibility advice even if the arithmetic had been checked. I would need to verify the applicable grading convention and the opportunity's own reporting and eligibility rules before using the number in a decision.

## Interpretation

The study documents a frequent larger-is-better default in the selected rubric-free tasks, limited recovery of the referenced convention from institution cues, and incomplete reliability with explicit keys. The sampled full-correctness comparison is tied. The current observations support a descriptive account of context and grading interpretation; they do not identify the training distribution or establish a general bias across models.

What stands out to me is how little the institution cues helped. In these tasks, Qwen3.5-4B frequently treated larger GPAs as better even when the prompt named the German University in Cairo. For Kymra, this means I would need evidence of the applicable grading convention before trusting the model’s assessment of academic eligibility. Recognizing a university’s name does not establish that the model understands its grading system.

The explicit-key results also changed how I would evaluate reliability. Full-response correctness was 50% for both keys, while key consistency was 55% for smaller-is-better and 70% for larger-is-better. These measure different things: an answer can follow the grading key and still make another substantive error. The sampled results therefore show a descriptive difference in key consistency, without a difference in full correctness. My takeaway is that calculations, grading interpretation and unsupported eligibility claims each need separate checks.

## Proposed remedy

I would fine-tune the evaluated checkpoint on a curated English–Arabic dataset built around matched examples. Student names, grades and credit weights would stay fixed while the supplied grading key changes. For example, 1.3 would represent stronger performance than 3.7 under a smaller-is-better key, and weaker performance under a larger-is-better key. Weighted-average examples would require both calculating the mean and interpreting it using the applicable convention. Examples with insufficient information would require clarification or clearly conditional reasoning.

I would also include institutional examples checked against official sources, with their program scope and policy date recorded. Constructed keys would be identified as hypothetical. The examples would distinguish a university’s grading rules from an opportunity’s application-reporting requirements: knowing which GPA is stronger does not establish whether an application requires conversion or a particular calculation.

These examples target the observed larger-is-better assumptions, inconsistent key use and unsupported reporting claims. The aim is to teach the model to ground its interpretation in the available evidence and recognize when essential information is missing. This intervention remains untested.

### How I would evaluate the remedy

I would create a fresh test set containing new scenarios and institutions excluded from the intervention’s training data. These cases would also be separate from the existing pilot and follow-up cores. Each scenario would have matched English and Arabic versions, with translations and paraphrases kept together to prevent overlap between training and testing. The set would cover both grading directions, missing-key cases, weighted averages and application-reporting questions.

I would compare the original and adapted checkpoints on identical prompts under matching quantization, thinking, decoding and token-budget settings. Both would receive a greedy evaluation and repeated sampled generations using the same seed schedule, fresh conversations and randomized run order. Expected answers and scoring rules would be frozen before generation, and reviewers would not be shown which checkpoint produced each answer.

I would score full correctness, key consistency, unsupported claims, appropriate clarification and required arithmetic separately. Clarification would be rewarded when essential information is missing, while unnecessary clarification would be tracked when sufficient information is supplied. Results would be reported by language and grading direction, with repeated outputs summarized within each scenario. I would look for improved correctness and key consistency, fewer unsupported claims, and appropriate handling of uncertainty without reduced arithmetic accuracy.

This evaluation used the original checkpoint. No fine-tuning or other remediation experiment was run as part of the reported study; any remedy above is a proposal unless accompanied by separately collected evidence.

## Limitations

- Eight selected cores, reused from a pilot, provide limited task diversity. Repeated draws do not create additional independent scenarios.
- Only one checkpoint, NF4 quantization configuration and thinking setting were evaluated. Results do not establish behavior of other models or full-precision deployments.
- The seven conditions are a limited comparison layout, not a complete language-by-institution-by-rubric factorial. Arabic institution cues and cue/rubric conflicts were not crossed.
- Arabic and Cairo do not uniquely identify a grading convention. An institution acronym can be ambiguous, and real policy applicability depends on the program and transcript.
- Full correctness includes the entire required response and substantive extra claims. Its rate should not be read as a pure measure of numerical ranking or arithmetic.
- The 14 token-limited outputs may reflect the response budget as well as grading interpretation. They are retained rather than selectively removed.
- These are constructed prompt tasks. Effects on real admissions decisions, students or registrar workflows were not measured.

## My contribution and assistance disclosure

I connected the grading-convention question to Kymra’s opportunity-matching problem and helped shape the study’s focus. I ran the prepared generation workflow, inspected the saved model responses and completed the human reviews in JSON. After repeated audits, I revisited flagged cases, revised my annotations and made the final decisions on ambiguous answers. I contributed the personal motivation and application-screening stakes, and reviewed the interpretation of the results, including the distinction between full correctness, key consistency and required arithmetic. AI assistance supported the technical setup and audits.

The follow-up was informed by the completed pilot. Later adjudication choices were made after inference and are reflected in the final annotations.

## Artifacts and reproduction

This package contains the completed follow-up dataset, all 348 saved responses, final human reviews, descriptive results and reproduction code. Model weights are available from the checkpoint and revision specified above.

| File | Contents |
| --- | --- |
| dataset/cases.csv | All 58 prompts, expected answers, criteria, condition labels and reference URLs |
| data/responses.jsonl | All 348 original model outputs, rendered inputs, token IDs, case seeds and finish reasons |
| data/reviews.csv | All 348 final human judgments and review notes |
| settings.json | Model revision, quantization, runtime versions, decoding settings and original call schedules |
| results/summary_metrics.json | Full descriptive results with per-prompt denominators and separate controls |
| results/condition_metrics.csv | Greedy and sampled results for the seven main conditions |
| results/figures/ | Four figures for grading direction and key consistency |
| analyze.py | CPU-only recalculation and verification of the saved summaries |
| generate.py and inference.py | Generation entry point and the model loader/generation implementation used in the study |
| notebooks/reproduce_generations.ipynb | Notebook for checking results and generating new outputs |
| requirements-gpu.txt | Pinned dependencies for GPU generation |

The block and case_id columns join responses and reviews to the dataset. Each line of responses.jsonl contains one response; input retains the rendered conversation and input token IDs, and output retains the original raw text, generated token IDs, seed and stopping condition. The reviews are final annotations.

With Python 3.11 or later, verify the existing results from the extracted package:

    python -B analyze.py --check

This uses only the Python standard library and requires no GPU or model download. To recalculate and save the summaries:

    python -B analyze.py --write

The generation schedule can also be checked without loading the model:

    python -B generate.py --blocks all

For a new GPU run, open the included notebook and follow its cells, or install requirements-gpu.txt in a compatible NVIDIA GPU environment and run:

    python -B generate.py --blocks all --generate --output ../grading_regenerated

The generation wrapper uses the recorded float16 compute setting, original model revision, fresh conversations, original per-case seed rule and saved call schedules. New responses are written separately and require new human reviews. Exact outputs may vary across software or hardware environments even with saved seeds. The notebook is an unexecuted reproduction entry point; the original outputs are the records in data/responses.jsonl.

The motivating references are [GUC General Study Regulations](https://www.guc.edu.eg/docs/content/policiesandregulations/general_study_regulations.pdf) and [MIU Grading System](https://www.miuegypt.edu.eg/admission-requirements/general-admission-requirments/admissions-regulations/grading-system/), recorded as accessed on 10 October 2026. Reference URLs are included with the cases. The supplied 0.0–5.0 experimental keys are constructed rubrics, not copies of those institutions' policies.
