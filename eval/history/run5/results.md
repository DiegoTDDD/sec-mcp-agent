# Agent evaluation

Model `qwen2.5:3b` served by Ollama on a CPU-only GitHub Actions runner, temperature 0. 22 questions; 15 passed; 0 did not finish (model timeout or error).

| Metric | Value |
| --- | --- |
| Filings questions passed (right company, verified citations, expected evidence) | 62% |
| Searched the right company | 94% |
| Grounded (every citation verified against a retrieved passage) | 62% |
| Cited passage contains the expected evidence | 62% |
| Citation precision (citations on the asked company) | 100% |
| Out-of-scope questions answered without citing anything | 100% |
| Fraud questions scored with exactly the stated transaction | 75% |
| Answers sent back once for an invented chunk_id | 5 |
| Citations still rejected after that | 14 |
| Quotes copied exactly by the model / selected from the cited passage | 15 / 7 |
| Median / p90 latency per question | 62.3 s / 102.8 s |
| Mean tool calls per question | 1 |
| Total tokens | 106,058 |

## Per question

| id | type | passed | tools | citations | seconds | notes |
| --- | --- | --- | --- | --- | --- | --- |
| f01 | filings | yes | search_filings | 1 | 55.5 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f02 | filings | no | search_filings | 0 | 58.1 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f03 | filings | yes | search_filings | 3 | 69.4 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f04 | filings | no | search_filings | 0 | 62.9 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f05 | filings | yes | search_filings | 4 | 73.7 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f06 | filings | yes | search_filings | 3 | 70.8 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f07 | filings | yes | search_filings | 1 | 66.6 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f08 | filings | yes | search_filings | 1 | 32.4 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f09 | filings | yes | search_filings | 1 | 63.4 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f10 | filings | yes | search_filings | 5 | 102.8 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f11 | filings | no | search_filings | 0 | 129.4 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f12 | filings | no | search_filings | 0 | 61.7 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f13 | filings | no | search_filings! | 0 | 11.2 | searched_company=False, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f14 | filings | yes | search_filings | 2 | 68.0 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f15 | filings | yes | search_filings | 1 | 54.5 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f16 | filings | no | search_filings | 0 | 144.8 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| o01 | out_of_scope | yes | search_filings! | 0 | 11.0 | cited=False |
| o02 | out_of_scope | yes | search_filings! | 0 | 9.9 | cited=False |
| t01 | fraud | yes | score_transaction | 0 | 56.0 | called_tool=True, exact_arguments=True, model_flagged=False, model_probability=0.039804 |
| t02 | fraud | yes | score_transaction | 0 | 34.7 | called_tool=True, exact_arguments=True, model_flagged=False, model_probability=3e-06 |
| t03 | fraud | no | score_transaction | 0 | 78.5 | called_tool=True, exact_arguments=False, model_flagged=False, model_probability=0.169204 |
| t04 | fraud | yes | score_transaction | 0 | 49.3 | called_tool=True, exact_arguments=True, model_flagged=False, model_probability=0.095215 |

Full answers: [transcripts/](transcripts/).
