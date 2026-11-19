# Agent evaluation

Model `qwen2.5:3b` served by Ollama on a CPU-only GitHub Actions runner, temperature 0. 22 questions; 11 passed; 0 did not finish (model timeout or error).

| Metric | Value |
| --- | --- |
| Filings questions passed (right company, verified citations, expected evidence) | 44% |
| Searched the right company | 100% |
| Grounded (every citation verified against a retrieved passage) | 50% |
| Cited passage contains the expected evidence | 44% |
| Citation precision (citations on the asked company) | 100% |
| Out-of-scope questions answered without citing anything | 50% |
| Fraud questions scored with exactly the stated transaction | 75% |
| Answers sent back once for an invented chunk_id | 4 |
| Citations still rejected after that | 17 |
| Quotes copied exactly by the model / selected from the cited passage | 22 / 5 |
| Median / p90 latency per question | 151.9 s / 288.7 s |
| Mean tool calls per question | 1.14 |
| Total tokens | 117,276 |

## Per question

| id | type | passed | tools | citations | seconds | notes |
| --- | --- | --- | --- | --- | --- | --- |
| f01 | filings | yes | search_filings | 1 | 151.8 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f02 | filings | no | search_filings | 0 | 124.1 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f03 | filings | yes | search_filings | 3 | 131.5 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f04 | filings | no | search_filings | 0 | 285.5 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f05 | filings | no | search_filings | 0 | 312.4 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f06 | filings | yes | search_filings | 4 | 190.0 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f07 | filings | yes | search_filings | 4 | 186.2 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f08 | filings | no | search_filings | 0 | 74.3 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f09 | filings | no | search_filings! search_filings | 1 | 105.7 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=False |
| f10 | filings | no | search_filings | 0 | 309.1 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f11 | filings | no | search_filings | 0 | 287.4 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f12 | filings | no | search_filings | 0 | 288.7 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f13 | filings | yes | search_filings | 5 | 185.2 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f14 | filings | no | search_filings | 0 | 240.9 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f15 | filings | yes | search_filings | 1 | 126.6 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f16 | filings | yes | search_filings | 3 | 151.9 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| o01 | out_of_scope | no | search_filings! search_filings | 5 | 250.6 | cited=True |
| o02 | out_of_scope | yes | search_filings! search_filings! | 0 | 62.2 | cited=False |
| t01 | fraud | yes | score_transaction | 0 | 83.1 | called_tool=True, exact_arguments=True, model_flagged=False, model_probability=0.039804 |
| t02 | fraud | yes | score_transaction | 0 | 89.2 | called_tool=True, exact_arguments=True, model_flagged=False, model_probability=3e-06 |
| t03 | fraud | no | score_transaction | 0 | 103.6 | called_tool=True, exact_arguments=False, model_flagged=False, model_probability=0.169204 |
| t04 | fraud | yes | score_transaction | 0 | 114.7 | called_tool=True, exact_arguments=True, model_flagged=False, model_probability=0.095215 |

Full answers: [transcripts/](transcripts/).
