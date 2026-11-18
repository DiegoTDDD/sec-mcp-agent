# Agent evaluation

Model `qwen2.5:3b` served by Ollama on a CPU-only GitHub Actions runner, temperature 0. 22 questions; 7 passed.

| Metric | Value |
| --- | --- |
| Filings questions passed (right company, verified citations, expected evidence) | 19% |
| Searched the right company | 38% |
| Grounded (every citation verified against a retrieved passage) | 44% |
| Cited passage contains the expected evidence | 31% |
| Citation precision (citations on the asked company) | 100% |
| Out-of-scope questions answered without citing anything | 100% |
| Fraud questions scored with exactly the stated transaction | 50% |
| Answers sent back once by the citation verifier | 9 |
| Citations still rejected after that | 10 |
| Median / p90 latency per question | 57.4 s / 176.1 s |
| Mean tool calls per question | 2.09 |
| Total tokens | 107,984 |

## Per question

| id | type | passed | tools | citations | seconds | notes |
| --- | --- | --- | --- | --- | --- | --- |
| f01 | filings | no | - | 0 | 44.6 | searched_company=False, grounded=True, cited=False, citations_on_target=None, evidence_found=False |
| f02 | filings | yes | search_filings! search_filings | 4 | 209.3 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f03 | filings | no | search_filings! search_filings! | 0 | 24.4 | searched_company=False, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f04 | filings | no | - | 0 | 28.3 | searched_company=False, grounded=True, cited=False, citations_on_target=None, evidence_found=False |
| f05 | filings | no | search_filings! search_filings! search_filings | 4 | 176.1 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=False |
| f06 | filings | no | search_filings! search_filings | 3 | 235.7 | searched_company=True, grounded=False, cited=True, citations_on_target=1.0, evidence_found=True |
| f07 | filings | no | - | 0 | 12.4 | searched_company=False, grounded=True, cited=False, citations_on_target=None, evidence_found=False |
| f08 | filings | no | search_filings! search_filings | 0 | 45.0 | searched_company=False, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f09 | filings | no | search_filings! search_filings! search_filings! get_chunk! score_transaction! | 0 | 69.6 | searched_company=False, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f10 | filings | no | search_filings! search_filings! search_filings! get_chunk! | 0 | 44.1 | searched_company=False, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f11 | filings | no | search_filings! search_filings! search_filings! | 0 | 37.8 | searched_company=False, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f12 | filings | no | search_filings! search_filings | 1 | 151.9 | searched_company=True, grounded=False, cited=True, citations_on_target=1.0, evidence_found=True |
| f13 | filings | no | search_filings! search_filings! search_filings! search_filings! | 0 | 45.9 | searched_company=False, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f14 | filings | yes | search_filings! search_filings | 2 | 272.1 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f15 | filings | no | search_filings! search_filings! | 0 | 32.2 | searched_company=False, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f16 | filings | yes | search_filings! search_filings | 2 | 110.6 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| o01 | out_of_scope | yes | search_filings! search_filings | 0 | 79.7 | cited=False |
| o02 | out_of_scope | yes | search_filings! search_filings! search_filings! search_filings! search_filings! | 0 | 64.7 | cited=False |
| t01 | fraud | yes | score_transaction | 0 | 100.2 | called_tool=True, exact_arguments=True, model_flagged=False, model_probability=0.039804 |
| t02 | fraud | no | score_transaction! | 0 | 34.4 | called_tool=False, exact_arguments=False, model_flagged=False, model_probability=3e-06 |
| t03 | fraud | no | score_transaction! | 0 | 50.0 | called_tool=False, exact_arguments=False, model_flagged=False, model_probability=0.169204 |
| t04 | fraud | yes | score_transaction | 0 | 83.8 | called_tool=True, exact_arguments=True, model_flagged=False, model_probability=0.095215 |

Full answers: [transcripts/](transcripts/).
