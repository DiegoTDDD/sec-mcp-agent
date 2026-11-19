# Agent evaluation

Model `qwen2.5:3b` served by Ollama on a CPU-only GitHub Actions runner, temperature 0. 22 questions; 11 passed; 1 did not finish (model timeout or error).

| Metric | Value |
| --- | --- |
| Filings questions passed (right company, verified citations, expected evidence) | 50% |
| Searched the right company | 38% |
| Grounded (every citation verified against a retrieved passage) | 56% |
| Cited passage contains the expected evidence | 50% |
| Citation precision (citations on the asked company) | 89% |
| Out-of-scope questions answered without citing anything | 50% |
| Fraud questions scored with exactly the stated transaction | 50% |
| Answers sent back once by the citation verifier | 11 |
| Citations still rejected after that | 5 |
| Median / p90 latency per question | 146.8 s / 211.4 s |
| Mean tool calls per question | 1.05 |
| Total tokens | 110,286 |

## Per question

| id | type | passed | tools | citations | seconds | notes |
| --- | --- | --- | --- | --- | --- | --- |
| f01 | filings | yes | search_filings | 1 | 156.5 | searched_company=False, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f02 | filings | yes | search_filings | 3 | 211.4 | searched_company=False, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f03 | filings | no | search_filings | 0 | 177.1 | searched_company=False, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f04 | filings | yes | search_filings | 4 | 154.1 | searched_company=False, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f05 | filings | no | search_filings | 0 | 123.2 | searched_company=False, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f06 | filings | yes | search_filings | 2 | 184.0 | searched_company=False, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f07 | filings | yes | search_filings | 4 | 142.3 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f08 | filings | no | search_filings | 0 | 75.0 | searched_company=False, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f09 | filings | no | search_filings! search_filings | 1 | 116.8 | searched_company=False, grounded=True, cited=True, citations_on_target=0.0, evidence_found=False |
| f10 | filings | no | search_filings | 0 | 174.3 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f11 | filings | no | search_filings | 0 | 115.5 | searched_company=False, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f12 | filings | yes | search_filings | 3 | 244.0 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f13 | filings | yes | search_filings | 4 | 182.1 | searched_company=True, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f14 | filings | yes | search_filings | 2 | 166.7 | searched_company=False, grounded=True, cited=True, citations_on_target=1.0, evidence_found=True |
| f15 | filings | no | search_filings | 0 | 151.3 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| f16 | filings | no | search_filings | 0 | 128.4 | searched_company=True, grounded=False, cited=False, citations_on_target=None, evidence_found=False |
| o01 | out_of_scope | yes | search_filings | 0 | 100.7 | cited=False |
| o02 | out_of_scope | no | - | 0 | 623.9 | error=ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception) |
| t01 | fraud | yes | score_transaction | 0 | 138.4 | called_tool=True, exact_arguments=True, model_flagged=False, model_probability=0.039804 |
| t02 | fraud | yes | score_transaction | 0 | 70.2 | called_tool=True, exact_arguments=True, model_flagged=False, model_probability=3e-06 |
| t03 | fraud | no | score_transaction! score_transaction | 0 | 124.9 | called_tool=True, exact_arguments=False, model_flagged=False, model_probability=0.169204 |
| t04 | fraud | no | score_transaction | 0 | 81.9 | called_tool=True, exact_arguments=False, model_flagged=False, model_probability=0.095215 |

Full answers: [transcripts/](transcripts/).
